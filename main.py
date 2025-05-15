import argparse
import cfg
import cv2
import numpy as np
import open3d as o3d
import torch
import numpy as np
import depth_pro
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from time import time


def load_depth_estimation_model():
    model, transform = depth_pro.create_model_and_transforms()
    model.to(cfg.DEVICE)
    model.eval()

    return model, transform 


def estimate_depth(model, image, f_px):
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"].cpu().numpy()
        focallength_px = prediction["focallength_px"].cpu().numpy()

    return depth, focallength_px


def run_object_detection(model, image, conf=0.4, imgsz=640):
    with torch.no_grad():
        results   = model.predict(
            image,
            conf=conf,
            imgsz=imgsz,
            )
        boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    return boxes, class_ids


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


def load_segmentation_model(model_type="vit_b", ckpt="./checkpoints/sam_vit_b_01ec64.pth"):
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=cfg.DEVICE)

    return SamPredictor(sam)


def get_object_mask(predictor, rgb_img, bbox):
    with torch.no_grad():
        x1, y1, x2, y2 = bbox
        predictor.set_image(rgb_img)
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)

    return masks[0]  # H×W, np.ndarray(bool)


def apply_mask_to_depth(depth_map, mask):
    masked = depth_map.copy()
    masked[~mask] = 0.0

    return masked


def depth_to_pointcloud(depth_region, focal_length=2600.0):
    # 워핑된 Depth의 해상도
    h, w = depth_region.shape

    # 픽셀 좌표 격자 생성
    u = np.linspace(0, w-1, w)
    v = np.linspace(0, h-1, h)
    uu, vv = np.meshgrid(u, v)

    # 3D 좌표 계산
    z = depth_region
    x = (uu - w/2) * z / focal_length
    y = (vv - h/2) * z / focal_length

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

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


def main(image_path, output_path, save_image=False):
    raw_img   = cv2.imread(image_path)
    rgb_img   = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    print("Current device:", cfg.DEVICE)

    print("Models Loading...")
    de_model, transform = load_depth_estimation_model()
    od_model = YOLO(cfg.OD_MODEL_PATH).to(cfg.DEVICE)
    sg_model = load_segmentation_model()

    stime = time()

    print("Depth Estimation running...")
    de_img, _, f_px = depth_pro.load_rgb(image_path)
    de_img = transform(de_img).to(cfg.DEVICE)
    depth_map, focal_length = estimate_depth(de_model, de_img, f_px)
    depth_map *= 100    # m -> cm

    print("Object Detection running...")
    boxes, class_ids = run_object_detection(od_model, rgb_img, conf=0.4, imgsz=640)

    print(f"Total processing time is {(time() - stime):.2f}s")

    depth_infos = []
    for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        cname = od_model.names[cls_id]
        
        # ▸ (1) 객체 마스크 얻기
        mask = get_object_mask(sg_model, rgb_img, box)

        # ▸ (2) 해당 박스 영역의 depth만 잘라서 마스킹
        x1, y1, x2, y2 = box
        rgb_crop = rgb_img[y1:y2, x1:x2]
        depth_crop = depth_map[y1:y2, x1:x2]
        mask_crop  = mask[y1:y2, x1:x2]
        masked_depth = apply_mask_to_depth(depth_crop, mask_crop)
        masked_rgb = apply_mask_to_depth(rgb_crop, mask_crop)

        # ▸ (3) PointCloud 변환 & 부피 계산
        points = depth_to_pointcloud(masked_depth, focal_length)  # bbox=None: 전체 사용
        volume, dims = calc_volume(points)

        info  = cfg.COCO_CLASS_INFO.get(cname, None)
        if info:
            avg_vol = info['width'] * info['height'] * info['depth']
            weight  = info['weight'] * (volume / avg_vol) if avg_vol > 0 else info['weight']
        else:
            weight = 0

        depth_infos.append({
            "id": i,
            "class": cname,
            "volume": volume,
            "weight": weight,
            "width":  dims[0],
            "height": dims[1],
            "depth":  dims[2]
        })

    # BBox 옆에 라벨 추가
    for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = box
        weight = depth_infos[i]['weight']
        pushable = "Pushable" if weight < 5000 else "Unpushable"
        
        label = f"{depth_infos[i]['class']}, Weight: {weight:.2f}g, {pushable}"
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
        main(args.image_path, output_path=args.output_path, save_image=args.save_image)
    # else:
    #     process_video(args.video_source)
