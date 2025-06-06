import os
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import time
import psutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LocalContrastNormalization:
    def __init__(self, kernel_size=9, epsilon=1e-6):
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img).astype(np.float32) / 255.0
        out = np.zeros_like(img_np)

        pil_blur = img.filter(ImageFilter.GaussianBlur(radius=self.kernel_size // 2))
        blur_np = np.array(pil_blur).astype(np.float32) / 255.0

        pil_blur_sq = img.filter(ImageFilter.GaussianBlur(radius=self.kernel_size // 2))
        blur_sq_np = np.array(pil_blur_sq).astype(np.float32) / 255.0

        sigma = np.sqrt(np.maximum(blur_sq_np - blur_np ** 2, self.epsilon))
        for c in range(3):
            out[..., c] = (img_np[..., c] - blur_np[..., c]) / (sigma[..., c] + self.epsilon)

        out = (out - out.min()) / (out.max() - out.min() + self.epsilon)
        out = (out * 255).astype(np.uint8)
        return Image.fromarray(out)

def prepare_camvid_dataset():
    dataset_dir = "CamVid"
    os.makedirs(dataset_dir, exist_ok=True)
    data_dir = os.path.join(dataset_dir, "data")

    if not os.path.exists(data_dir):
        print("Downloading CamVid dataset")
        url = "https://github.com/alexgkendall/SegNet-Tutorial/archive/refs/heads/master.zip"
        import requests, zipfile, shutil
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open("camvid_temp.zip", 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size//1024, unit='KB', desc="Downloading CamVid"):
                f.write(chunk)
        print("Extracting CamVid dataset")
        with zipfile.ZipFile("camvid_temp.zip", 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        src = os.path.join(dataset_dir, "SegNet-Tutorial-master", "CamVid")
        os.rename(src, data_dir)
        os.remove("camvid_temp.zip")
        shutil.rmtree(os.path.join(dataset_dir, "SegNet-Tutorial-master"))
        print("CamVid dataset is ready")
    return data_dir

class CamVidDataset(Dataset):
    def __init__(self, root, split='train', image_size=(360, 480)):
        super().__init__()
        self.split = split
        self.root = root
        self.image_size = image_size

        list_file = os.path.join(root, f"{split}.txt")
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Cannot find list file: {list_file}")

        with open(list_file, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        self.pairs = []
        missing = 0
        for ln in lines:
            tokens = ln.split()
            if len(tokens) == 1:
                img_name = os.path.basename(tokens[0])
                lbl_name = img_name.replace('.png', '_L.png')
            else:
                img_name = os.path.basename(tokens[0])
                lbl_name = os.path.basename(tokens[1])

            img_path = os.path.join(root, split, img_name)
            lbl_path = os.path.join(root, split + 'annot', lbl_name)
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                self.pairs.append((img_path, lbl_path))
            else:
                missing += 1
        print(f"Found {len(self.pairs)} pairs in '{split}' (skipped {missing})")

        self.lcn = LocalContrastNormalization()
        self.color_jitter = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path)

        if self.split == 'train':
            image = self.lcn(image)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.5, 2.0), ratio=(1.0, 1.0)
            )
            image = TF.resized_crop(image, i, j, h, w, self.image_size, interpolation=Image.BILINEAR)
            label = TF.resized_crop(label, i, j, h, w, self.image_size, interpolation=Image.NEAREST)
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            image = self.color_jitter(image)
        else:
            image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
            label = TF.resize(label, self.image_size, interpolation=Image.NEAREST)

        image = self.to_tensor(image)
        image = self.normalize(image)

        label_np = np.array(label, dtype=np.int64)
        label_np[label_np == 11] = 255
        label_tensor = torch.from_numpy(label_np).long()

        return image, label_tensor

def calculate_global_accuracy(preds, labels):
    pred_labels = preds.argmax(dim=1)
    valid = (labels != 255)
    correct = ((pred_labels == labels) & valid).sum().item()
    total = valid.sum().item()
    return correct / total if total > 0 else 0.0


def calculate_miou(preds, labels, num_classes=11):
    pred_labels = preds.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred_labels == cls)
        true_mask = (labels == cls)
        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0

def train_transformer_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=100):
    best_miou = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    train_history = {'loss': [], 'global_acc': [], 'miou': []}
    val_history = {'loss': [], 'global_acc': [], 'miou': []}
    best_class_ious = None
    start_time = time.time()
    memory_usage = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_miou = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_acc += calculate_global_accuracy(outputs, masks)
            running_miou += calculate_miou(outputs, masks, num_classes=11)
            memory_usage.append(psutil.virtual_memory().used / (1024 ** 3))

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        epoch_miou = running_miou / len(train_loader)
        train_history['loss'].append(epoch_loss)
        train_history['global_acc'].append(epoch_acc)
        train_history['miou'].append(epoch_miou)
        print(f"Train: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, mIoU: {epoch_miou:.4f}")

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_miou = 0.0
        per_class_ious = torch.zeros(11)
        total_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val  ]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(pixel_values=images).logits
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_acc += calculate_global_accuracy(outputs, masks)
                val_miou += calculate_miou(outputs, masks, num_classes=11)

                pred_labels = outputs.argmax(dim=1)
                for cls in range(11):
                    pred_mask = (pred_labels == cls)
                    true_mask = (masks == cls)
                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()
                    if union > 0:
                        per_class_ious[cls] += intersection / union
                total_batches += 1

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_acc / len(val_loader)
        epoch_val_miou = val_miou / len(val_loader)
        val_history['loss'].append(epoch_val_loss)
        val_history['global_acc'].append(epoch_val_acc)
        val_history['miou'].append(epoch_val_miou)
        print(f"Val: Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, mIoU: {epoch_val_miou:.4f}")

        if epoch_val_miou > best_miou:
            best_miou = epoch_val_miou
            best_class_ious = (per_class_ious / total_batches).cpu().numpy()
            torch.save(model.state_dict(), "best_segformer_camvid.pth")

    total_time = time.time() - start_time
    avg_memory = np.mean(memory_usage)
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Average memory usage: {avg_memory:.2f} GB")

    print(f"Best Val mIoU: {best_miou:.4f}")
    return train_history, val_history, best_class_ious

def test_transformer_model(model, test_loader, device):
    model.eval()
    test_acc = 0.0
    test_miou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(
                outputs, 
                size=masks.shape[1:], 
                mode='bilinear', 
                align_corners=False
            )

            test_acc += calculate_global_accuracy(outputs, masks)
            test_miou += calculate_miou(outputs, masks, num_classes=11)

    avg_acc  = test_acc / len(test_loader)
    avg_miou = test_miou / len(test_loader)
    print(f"Test Results: Acc: {avg_acc:.4f}, mIoU: {avg_miou:.4f}")

def visualize_results(train_history, val_history, class_ious):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_history['global_acc'], label='Train Global Acc')
    plt.plot(val_history['global_acc'], label='Val Global Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(train_history['miou'], label='Train mIoU')
    plt.plot(val_history['miou'], label='Val mIoU')
    plt.title('mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()

    plt.subplot(2, 2, 4)
    classes = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole',
               'Traffic Light', 'Traffic Sign', 'Vegetation', 'Terrain', 'Sky']
    plt.bar(classes, class_ious.cpu().numpy())
    plt.title('Per-Class IoU')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('IoU')
    plt.tight_layout()

    plt.show()


data_dir = prepare_camvid_dataset()

image_size = (360, 480)
batch_size = 8
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = CamVidDataset(data_dir, split='train', image_size=image_size)
val_ds = CamVidDataset(data_dir, split='val', image_size=image_size)
test_ds = CamVidDataset(data_dir, split='test', image_size=image_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    ignore_mismatched_sizes=True,
    num_labels=11,
    id2label={str(i): str(i) for i in range(11)},
    label2id={str(i): i for i in range(11)}
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


train_history, val_history, class_ious = train_transformer_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=num_epochs)
visualize_results(train_history, val_history, class_ious)

test_transformer_model(model, test_loader, device)