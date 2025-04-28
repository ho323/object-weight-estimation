import random
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOCAL_LENGTH = {
    'fx': 26,
    'fy': 26,
}


# ================== 물체 인식 모델 ================== #
# COCO 클래스 목록
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COCO_CLASS_INFO = {
    'person': {'weight': 70, 'width': 50, 'height': 170, 'depth': 30},
    'bicycle': {'weight': 15, 'width': 50, 'height': 100, 'depth': 180},
    'car': {'weight': 1500, 'width': 180, 'height': 150, 'depth': 450},
    'motorcycle': {'weight': 200, 'width': 80, 'height': 110, 'depth': 200},
    'airplane': {'weight': 40000, 'width': 2800, 'height': 600, 'depth': 3000},
    'bus': {'weight': 12000, 'width': 250, 'height': 320, 'depth': 1200},
    'train': {'weight': 200000, 'width': 320, 'height': 450, 'depth': 2500},
    'truck': {'weight': 10000, 'width': 250, 'height': 350, 'depth': 800},
    'boat': {'weight': 500, 'width': 200, 'height': 150, 'depth': 500},
    'traffic light': {'weight': 100, 'width': 30, 'height': 100, 'depth': 30},
    'fire hydrant': {'weight': 75, 'width': 40, 'height': 80, 'depth': 40},
    'stop sign': {'weight': 5, 'width': 60, 'height': 60, 'depth': 1},
    'parking meter': {'weight': 30, 'width': 30, 'height': 120, 'depth': 30},
    'bench': {'weight': 50, 'width': 50, 'height': 80, 'depth': 150},
    'bird': {'weight': 0.1, 'width': 10, 'height': 15, 'depth': 10},
    'cat': {'weight': 4, 'width': 20, 'height': 30, 'depth': 50},
    'dog': {'weight': 10, 'width': 30, 'height': 50, 'depth': 80},
    'horse': {'weight': 400, 'width': 80, 'height': 160, 'depth': 200},
    'sheep': {'weight': 80, 'width': 60, 'height': 80, 'depth': 100},
    'cow': {'weight': 600, 'width': 80, 'height': 150, 'depth': 220},
    'elephant': {'weight': 5000, 'width': 150, 'height': 300, 'depth': 600},
    'bear': {'weight': 300, 'width': 100, 'height': 120, 'depth': 200},
    'zebra': {'weight': 350, 'width': 60, 'height': 140, 'depth': 250},
    'giraffe': {'weight': 1200, 'width': 100, 'height': 550, 'depth': 200},
    'backpack': {'weight': 3, 'width': 30, 'height': 50, 'depth': 20},
    'umbrella': {'weight': 0.5, 'width': 10, 'height': 100, 'depth': 10},
    'handbag': {'weight': 2, 'width': 30, 'height': 30, 'depth': 15},
    'tie': {'weight': 0.2, 'width': 10, 'height': 70, 'depth': 1},
    'suitcase': {'weight': 5, 'width': 40, 'height': 60, 'depth': 20},
    'frisbee': {'weight': 0.2, 'width': 25, 'height': 3, 'depth': 25},
    'skis': {'weight': 4, 'width': 10, 'height': 5, 'depth': 180},
    'snowboard': {'weight': 5, 'width': 30, 'height': 10, 'depth': 150},
    'sports ball': {'weight': 0.5, 'width': 22, 'height': 22, 'depth': 22},
    'kite': {'weight': 0.2, 'width': 100, 'height': 1, 'depth': 100},
    'baseball bat': {'weight': 1, 'width': 7, 'height': 7, 'depth': 100},
    'baseball glove': {'weight': 0.6, 'width': 25, 'height': 25, 'depth': 15},
    'skateboard': {'weight': 3, 'width': 20, 'height': 10, 'depth': 80},
    'surfboard': {'weight': 10, 'width': 60, 'height': 15, 'depth': 250},
    'tennis racket': {'weight': 0.3, 'width': 30, 'height': 3, 'depth': 70},
    'bottle': {'weight': 1, 'width': 8, 'height': 25, 'depth': 8},
    'wine glass': {'weight': 0.4, 'width': 8, 'height': 20, 'depth': 8},
    'cup': {'weight': 0.3, 'width': 8, 'height': 10, 'depth': 8},
    'fork': {'weight': 0.1, 'width': 3, 'height': 2, 'depth': 20},
    'knife': {'weight': 0.2, 'width': 3, 'height': 2, 'depth': 30},
    'spoon': {'weight': 0.1, 'width': 4, 'height': 2, 'depth': 20},
    'bowl': {'weight': 0.5, 'width': 15, 'height': 8, 'depth': 15},
    'banana': {'weight': 0.2, 'width': 5, 'height': 5, 'depth': 20},
    'apple': {'weight': 0.25, 'width': 8, 'height': 8, 'depth': 8},
    'sandwich': {'weight': 0.3, 'width': 10, 'height': 5, 'depth': 15},
    'orange': {'weight': 0.3, 'width': 8, 'height': 8, 'depth': 8},
    'broccoli': {'weight': 0.4, 'width': 15, 'height': 20, 'depth': 15},
    'carrot': {'weight': 0.15, 'width': 3, 'height': 3, 'depth': 20},
    'hot dog': {'weight': 0.25, 'width': 5, 'height': 5, 'depth': 15},
    'pizza': {'weight': 0.5, 'width': 30, 'height': 2, 'depth': 30},
    'donut': {'weight': 0.1, 'width': 10, 'height': 5, 'depth': 10},
    'cake': {'weight': 1, 'width': 20, 'height': 10, 'depth': 20},
    'chair': {'weight': 7, 'width': 50, 'height': 90, 'depth': 50},
    'couch': {'weight': 40, 'width': 200, 'height': 90, 'depth': 80},
    'potted plant': {'weight': 5, 'width': 30, 'height': 60, 'depth': 30},
    'bed': {'weight': 50, 'width': 160, 'height': 50, 'depth': 200},
    'dining table': {'weight': 30, 'width': 150, 'height': 75, 'depth': 100},
    'toilet': {'weight': 25, 'width': 40, 'height': 70, 'depth': 70},
    'tv': {'weight': 8, 'width': 80, 'height': 50, 'depth': 3},
    'laptop': {'weight': 2, 'width': 35, 'height': 2.5, 'depth': 25},
    'mouse': {'weight': 0.1, 'width': 6, 'height': 3, 'depth': 10},
    'remote': {'weight': 0.2, 'width': 5, 'height': 2, 'depth': 20},
    'keyboard': {'weight': 1.2, 'width': 45, 'height': 2, 'depth': 15},
    'cell phone': {'weight': 0.2, 'width': 8, 'height': 1, 'depth': 15},
    'microwave': {'weight': 15, 'width': 50, 'height': 40, 'depth': 50},
    'oven': {'weight': 40, 'width': 80, 'height': 90, 'depth': 60},
    'toaster': {'weight': 3, 'width': 30, 'height': 20, 'depth': 30},
    'sink': {'weight': 20, 'width': 60, 'height': 25, 'depth': 50},
    'refrigerator': {'weight': 70, 'width': 80, 'height': 180, 'depth': 70},
    'book': {'weight': 0.5, 'width': 15, 'height': 2, 'depth': 20},
    'clock': {'weight': 1, 'width': 30, 'height': 30, 'depth': 5},
    'vase': {'weight': 2, 'width': 20, 'height': 30, 'depth': 20},
    'scissors': {'weight': 0.2, 'width': 6, 'height': 2, 'depth': 20},
    'teddy bear': {'weight': 1, 'width': 40, 'height': 50, 'depth': 30},
    'hair drier': {'weight': 0.8, 'width': 20, 'height': 15, 'depth': 25},
    'toothbrush': {'weight': 0.1, 'width': 2, 'height': 2, 'depth': 20}
}


random.seed(42)
CLASS_COLORS = [(random.random(), random.random(), random.random()) for _ in range(len(COCO_CLASSES))]
OD_MODEL_PATH = "./checkpoints/yolov8l-seg.pt"


# ================== 깊이 추정 모델 ================== #
DE_MODEL_PATH = './checkpoints/depth_anything_v2_vits.pth'


# ================== 재료 분류 모델 ================== #
MATARIALS_MODEL_PATH = "./checkpoints/material_classification_efficientnet_minc.pth" 
MTR_CATEGORY = [
    "brick",
    "carpet",
    "ceramic",
    "fabric",
    "foliage",
    "food",
    "glass",
    "hair",
    "leather",
    "metal",
    "mirror",
    "other",
    "painted",
    "paper",
    "plastic",
    "polishedstone",
    "skin",
    "sky",
    "stone",
    "tile",
    "wallpaper",
    "water",
    "wood",
    ]

# 밀도 (g/cm³)
DENSITIES = {
        "brick": 1.8,
        "carpet": 0.3,
        "ceramic": 2.4,
        "fabric": 0.6,
        "foliage": 0.5,
        "food": 1.0,
        "glass": 2.5,
        "hair": 0.03,
        "leather": 0.9,
        "metal": 7.8,
        "mirror": 2.5,
        "other": 1.0,
        "painted": 1.3,
        "paper": 0.8,
        "plastic": 1.2,
        "polishedstone": 2.7,
        "skin": 1.1,
        "sky": 0.0012,  #공기 
        "stone": 2.7,
        "tile": 2.0,
        "wallpaper": 0.7,
        "water": 1.0,
        "wood": 0.6
    }