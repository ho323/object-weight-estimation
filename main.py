import argparse
import cfg
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from model.yolo import load_yolo_model, run_yolo_inference
from depth_anything_v2.dpt import DepthAnythingV2
from time import time


# Depth Estimation을 위한 함수
def load_depth_model(device):
    de_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    de_model.load_state_dict(torch.load(cfg.DE_MODEL_PATH, map_location=device))
    de_model.to(device)
    de_model.eval()
    return de_model

def estimate_depth(model, raw_img):
    with torch.no_grad():
        return model.infer_image(raw_img)


def extract_bounding_boxes_and_crops(results):
    result = results[0]  # Assuming results are in a list
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # 클래스 ID들

    cropped_images = []
    class_names = []

    for box, cls_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        crop = result.orig_img[y1:y2, x1:x2].copy()
        cropped_images.append(crop)

        class_name = result.names[cls_id]
        class_names.append(class_name)

    return cropped_images, boxes, class_ids, class_names

# Depth to PointCloud 및 부피 계산 함수
def depth_to_pointcloud(depth_map, bbox):
    x_min, y_min, x_max, y_max = bbox
    fx = cfg.CAMERA_PARAMETER['fx']
    fy = cfg.CAMERA_PARAMETER['fy']
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    print(cfg.CAMERA_PARAMETER)

    depth_region = depth_map[y_min:y_max, x_min:x_max]
    h, w = depth_region.shape
    print(h, w)

    u = np.linspace(x_min, x_max - 1, w)
    v = np.linspace(y_min, y_max - 1, h)
    uu, vv = np.meshgrid(u, v)

    # 3D 좌표 계산 (벡터화된 방식)
    x = (uu - cx) * depth_region / fx
    y = (vv - cy) * depth_region / fy
    z = depth_region

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def calc_volume(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    voxel_size = 1  # Voxel 크기 (1cm)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    num_voxels = len(voxel_grid.get_voxels())
    voxel_volume = voxel_size ** 3
    volume = num_voxels * voxel_volume

    # 물체의 크기 계산
    points_array = np.asarray(pcd.points)
    min_bound = np.min(points_array, axis=0)
    max_bound = np.max(points_array, axis=0)
    dimensions = max_bound - min_bound  # [width, height, depth]

    return volume, dimensions

# 물질 분류 및 무게 계산 함수
def weight_estimation(cropped_images, boxes, class_ids, depth_map, result):
    out = []
    for i, (img, box, cls_id) in enumerate(zip(cropped_images, boxes, class_ids)):
        points = depth_to_pointcloud(depth_map, box)
        volume, dimensions = calc_volume(points)

        cname = result.names[cls_id]
        info  = cfg.COCO_CLASS_INFO.get(cname)
        if info:
            avg_vol = info['width'] * info['height'] * info['depth']
            weight  = info['weight'] * (volume / avg_vol) if avg_vol > 0 else info['weight']
        else:
            weight = 0

        out.append({
            "id": i, 
            "class": cname, 
            "volume": volume, 
            "weight": weight,
            "width": dimensions[0],
            "height": dimensions[1], 
            "depth": dimensions[2]
        })
    return out

# 전체 파이프라인 함수
def process_image(image_path, output_path, save_image=False):
    raw_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # 이미지 변환 (BGR -> RGB)

    print("Current device:", cfg.DEVICE)

    print("Models Loading...")
    de_model = load_depth_model(cfg.DEVICE)
    od_model = load_yolo_model(cfg.OD_MODEL_PATH, cfg.DEVICE)
    
    print("Depth Estimation running...")
    depth_map = estimate_depth(de_model, rgb_img)

    print("YOLOv8 running...")
    results = run_yolo_inference(od_model, raw_img, conf_threshold=0.7)
    cropped_images, boxes, class_ids, class_names = extract_bounding_boxes_and_crops(results)

    print("Weight Estimation running...")
    stime = time()
    depth_infos = weight_estimation(cropped_images, boxes, class_ids, depth_map, results[0])
    print(f"Total processing time is {(time() - stime):.2f}s")

    # BBox 옆에 라벨 추가
    for i, (box, cls_id, class_name) in enumerate(zip(boxes, class_ids, class_names)):
        x1, y1, x2, y2 = box
        weight = depth_infos[i]['weight']
        pushable = "Pushable" if weight < 3.5 else "Unpushable"
        
        label = f"{class_name}, Weight: {weight:.1f}kg, {pushable}"
        print(label)
        
        # BBox 그리기
        if pushable == "Pushable":
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # 라벨 추가
        cv2.putText(rgb_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (125, 255, 125), 3)

    if save_image:
        # RGB 이미지를 저장 (BGR로 변환하여 저장)
        output_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 다시 BGR로 변환
        cv2.imwrite(output_path, output_bgr)
        print("Image saved as output.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO and Depth Estimation on Image or Video")
    parser.add_argument('--image_path', type=str, help="Path to the input image")
    parser.add_argument('--output_path', default="output.png" ,type=str, help="Path to the input image")
    parser.add_argument('--video_source', type=str, default="0", help="Path to the video file or camera source")
    parser.add_argument('--save_image', action='store_true', help="Flag to save output image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image_path:
        process_image(args.image_path, output_path=args.output_path, save_image=args.save_image)
    # else:
    #     process_video(args.video_source)
