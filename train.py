"""
author :Samil SAYEDALİ
Iris Flower Image Classification with CNN and Sharpness-Aware Minimization (SAM)

Expected dataset layout:
dataset/
    iris-setosa/
        image1.jpg
        ...
    iris-versicolor/
        ...
    iris-virginica/
        ...

Run:
    python train.py --data_dir ./dataset --epochs 30
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SAM:
    """A practical two-pass SAM wrapper around a PyTorch optimizer."""

    def __init__(self, params, base_optimizer=torch.optim.AdamW,
                 rho: float = 0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norms = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not grad_norms:
            return

        grad_norm = torch.norm(torch.stack(grad_norms), p=2)
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                perturbation = p.grad * scale.to(p)
                p.add_(perturbation)
                self._state(p)["e_w"] = perturbation

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                state = self._state(p)
                if "e_w" in state:
                    p.sub_(state.pop("e_w"))

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _state(self, parameter):
        if not hasattr(self, "_perturbation_state"):
            self._perturbation_state = {}
        key = id(parameter)
        if key not in self._perturbation_state:
            self._perturbation_state[key] = {}
        return self._perturbation_state[key]


class IrisCNN(nn.Module):
    """Compact CNN that adapts its output layer to the dataset classes."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_loaders(data_dir: Path, image_size: int, batch_size: int,
                  seed: int, num_workers: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # Separate dataset objects allow augmentation only on the training subset.
    train_full = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    eval_full = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)

    if len(train_full.classes) < 2:
        raise ValueError("The dataset must contain at least two class folders.")

    targets = np.asarray(train_full.targets)
    indices = np.arange(len(targets))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=targets
    )
    temp_targets = targets[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=temp_targets
    )

    train_targets = targets[train_idx]
    class_counts = np.bincount(train_targets, minlength=len(train_full.classes))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    common = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        Subset(train_full, train_idx), sampler=sampler, **common
    )
    val_loader = DataLoader(
        Subset(eval_full, val_idx), shuffle=False, **common
    )
    test_loader = DataLoader(
        Subset(eval_full, test_idx), shuffle=False, **common
    )
    return train_loader, val_loader, test_loader, train_full.classes


def run_epoch(model, loader, criterion, device, optimizer=None, is_sam=False):
    training = optimizer is not None
    model.train(training)
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if is_sam:
                optimizer.first_step(zero_grad=True)
                second_logits = model(images)
                second_loss = criterion(second_logits, labels)
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, np.asarray(all_labels), np.asarray(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Train a CNN with SAM on iris images.")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Folder containing one subfolder per flower class.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = build_loaders(
        args.data_dir, args.image_size, args.batch_size,
        args.seed, args.num_workers
    )
    print(f"Classes: {class_names}")
    print(f"Split sizes — train: {len(train_loader.dataset)}, "
          f"validation: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    model = IrisCNN(num_classes=len(class_names)).to(device)
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=args.rho,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_accuracy": [],
               "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, is_sam=True
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.3%} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.3%}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
            }, args.output_dir / "best_model.pth")

    # Evaluate the best checkpoint on the held-out test set.
    checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = run_epoch(
        model, test_loader, criterion, device
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.3%}")
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, zero_division=0
    ))

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "classification_report": report,
            "history": history,
        }, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=30)
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"\nArtifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
