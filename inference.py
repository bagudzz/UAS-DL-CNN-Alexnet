import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms
from PIL import Image


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def load_model(weights_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(weights_path, map_location=device)

    model = alexnet(weights=AlexNet_Weights.DEFAULT)
    in_f = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_f, 2)

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    class_to_idx = ckpt.get("class_to_idx", {"cat": 0, "dog": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class, device


@torch.no_grad()
def predict(pil_img: Image.Image, model, idx_to_class, device):
    x = get_preprocess()(pil_img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    return {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
