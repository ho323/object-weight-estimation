import argparse
import cfg
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from model.yolo import load_yolo_model, run_yolo_on_image
from model.mtr_clf import init_model, materials_classification
from depth_anything_v2.dpt import DepthAnythingV2
from time import time


# Depth Estimation을 위한 함수
def load_depth_model(device):
    de_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    de_model.load_state_dict(torch.load(cfg.DE_MODEL_PATH, map_location=device))
    de_model.to(device)
    de_model.eval()
    return de_model

def estimate_depth(raw_img, model, device):
    with torch.no_grad():
        return model.infer_image(raw_img)

# YOLO 모델을 위한 함수
def load_yolo_model_and_run(raw_img, model_path, conf_threshold=0.6):
    model = load_yolo_model(model_path)
    results = run_yolo_on_image(raw_img, model, conf_threshold)
    return results

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
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    depth_region = depth_map[y_min:y_max, x_min:x_max]

    u, v = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

    # 3D 좌표 계산 (벡터화된 방식)
    x = (u - cx) * depth_region / cfg.FOCAL_LENGTH['fx']
    y = (v - cy) * depth_region / cfg.FOCAL_LENGTH['fy']
    z = depth_region

    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    return points, x, y, z  # x, y, z를 반환

def calc_volume(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    voxel_size = 1  # Voxel 크기 (1cm)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    num_voxels = len(voxel_grid.get_voxels())
    voxel_volume = voxel_size ** 3
    volume = num_voxels * voxel_volume

    return volume

# 물질 분류 및 무게 계산 함수
def material_classification_and_weight(cropped_images, boxes, class_ids, depth_map, mtr_model, mtr_classes, result):
    depth_infos = []
    for i, (crop_img, box, cls_id) in enumerate(zip(cropped_images, boxes, class_ids)):
        # 3D 좌표 및 가로, 세로, 깊이 정보 얻기
        points, x, y, z = depth_to_pointcloud(depth_map, box)

        # 가로, 세로 계산 (BBox 영역에서의 x와 y의 차이를 계산)
        width = np.max(x) - np.min(x)  # x의 최대/최소 차이로 가로
        height = np.max(y) - np.min(y)  # y의 최대/최소 차이로 세로
        depth = np.mean(z)  # 평균 깊이

        # 부피 계산
        volume = calc_volume(points)

        image_pil = Image.fromarray(crop_img)
        material, _ = materials_classification(mtr_model, mtr_classes, image_pil)

        depth_infos.append({
            "id": i,
            "class": result.names[cls_id],
            "volume": volume,
            "material": material,
            "width": width,  
            "height": height,  
            "depth": depth,
        })

    return depth_infos

def calculate_weight_for_materials(depth_infos):
    for info in depth_infos:
        material = info['material']
        weight = cfg.DENSITIES[material] * info['volume']
        print(f"[{info['id']}] {info['class']}'s Weight: {weight/1000:.2f}kg")

# 전체 파이프라인 함수
def process_image(image_path, output_path, save_image=False):
    raw_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # 이미지 변환 (BGR -> RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Depth Estimation running...")
    de_model = load_depth_model(device)
    
    print("YOLOv8 running...")
    results = load_yolo_model_and_run(rgb_img, cfg.OD_MODEL_PATH, conf_threshold=0.6)

    cropped_images, boxes, class_ids, class_names = extract_bounding_boxes_and_crops(results)

    mtr_model, mtr_classes = init_model()
    depth_infos = material_classification_and_weight(cropped_images, boxes, class_ids, de_model.infer_image(rgb_img), mtr_model, mtr_classes, results[0])

    # BBox 옆에 라벨 추가
    for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = box
        material = depth_infos[i]['material']
        volume = depth_infos[i]['volume']
        weight = cfg.DENSITIES[material] * volume
        width = depth_infos[i]['width']
        height = depth_infos[i]['height']
        depth = depth_infos[i]['depth']
        
        label = f"{results[0].names[cls_id]} {material} Vol: {volume:.2f}cm³ W: {weight/1000:.2f}kg"
        dimension_info = f"WxH: {width:.2f}x{height:.2f}cm D: {depth:.2f}cm"
        
        # BBox 그리기
        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 라벨 추가
        cv2.putText(rgb_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(rgb_img, dimension_info, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if save_image:
        # RGB 이미지를 저장 (BGR로 변환하여 저장)
        output_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 다시 BGR로 변환
        cv2.imwrite(output_path, output_bgr)
        print("Image saved as output.png")


# 실시간 영상 또는 동영상 처리 함수
def process_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Cannot open camera or video.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    de_model = load_depth_model(device)
    mtr_model, mtr_classes = init_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print("Depth Estimation running...")
        depth_map = estimate_depth(raw_img, de_model, device)

        print("YOLOv8 running...")
        results = load_yolo_model_and_run(frame, cfg.OD_MODEL_PATH, conf_threshold=0.6)

        cropped_images, boxes, class_ids, class_names = extract_bounding_boxes_and_crops(results)

        # 물질 분류 및 무게 계산
        depth_infos = material_classification_and_weight(cropped_images, boxes, class_ids, depth_map, mtr_model, mtr_classes, results[0])

        # 물체별 정보 및 BBox 그리기
        for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
            x1, y1, x2, y2 = box
            material = depth_infos[i]['material']
            volume = depth_infos[i]['volume']
            weight = cfg.DENSITIES[material] * volume
            width = depth_infos[i]['width']
            height = depth_infos[i]['height']
            depth = depth_infos[i]['depth']
            
            label = f"{results[0].names[cls_id]} {material} Vol: {volume:.2f}cm³ W: {weight/1000:.2f}kg"
            dimension_info = f"WxH: {width:.2f}x{height:.2f}cm D: {depth:.2f}cm"
            
            # BBox 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 라벨 추가
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, dimension_info, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 실시간 화면에 출력
        cv2.imshow('Real-Time Object Detection', frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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
    else:
        process_video(args.video_source)
