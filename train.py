import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights


def build_dataloaders(train_dir: str, test_dir: str, batch_size: int = 32):
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(train_dir, transform=tfm)
    test_ds = ImageFolder(test_dir, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_ds, test_ds, train_dl, test_dl


def build_model(num_classes=2, freeze_features=True):
    model = alexnet(weights=AlexNet_Weights.DEFAULT)

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False

    in_f = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_f, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / max(total, 1)


def main():
    # Ubah ini sesuai folder dataset kamu
    train_dir = "data/train"
    test_dir = "data/test"
    out_path = "weights/alexnet_catsdogs.pth"

    epochs = 5
    batch_size = 32
    lr = 1e-3  # jauh lebih masuk akal dari 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("weights", exist_ok=True)

    train_ds, test_ds, train_dl, test_dl = build_dataloaders(train_dir, test_dir, batch_size)
    print("class_to_idx:", train_ds.class_to_idx)  # penting untuk mapping label

    model = build_model(num_classes=2, freeze_features=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)

        loss_epoch = running / max(len(train_ds), 1)
        acc = evaluate(model, test_dl, device)
        print(f"Epoch {epoch}: loss={loss_epoch:.4f} | test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    torch.save({"model_state": best_state, "class_to_idx": train_ds.class_to_idx}, out_path)
    print(f"Saved: {out_path} (best_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
