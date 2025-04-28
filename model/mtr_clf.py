import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import os
import cfg

# === 환경 설정 ===
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "MINC_IMAGENET1K_V1.pth"  # 학습된 모델 경로
# CATEGORY_FILE = "./minc/minc-2500/categories.txt"  # 클래스 리스트 경로

def init_model():
    # === 클래스 목록 ===
    classes = cfg.MTR_CATEGORY

    # === 모델 정의 및 weight 로드 ===
    model = efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(cfg.MATARIALS_MODEL_PATH, map_location=cfg.DEVICE))
    model.eval().to(cfg.DEVICE)

    return model, classes

# === 단일 이미지 추론 함수 ===
def materials_classification(model, classes, image):
    # === 추론용 전처리 정의 ===
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, pred = torch.max(probs, dim=1)

    predicted_class = classes[pred.item()]
    confidence = top_prob.item()
    return predicted_class, confidence

def main():
    test_image = "./minc/minc-2500/images/brick/brick_001981.jpg"  # 테스트할 이미지 경로
    if not os.path.exists(test_image):
        print("❗ 테스트 이미지가 존재하지 않습니다:", test_image)
    else:
        model, classes = init_model()
        image = Image.open(test_image).convert("RGB")
        label, conf = materials_classification(model, classes, image)
        print(f"🔍 예측 결과: {label} ({conf*100:.2f}%)")

# === 테스트 예시 ===
if __name__ == "__main__":
    main()
