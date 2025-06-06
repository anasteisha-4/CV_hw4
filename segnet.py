import os
import time
import shutil
import requests
import zipfile
import psutil
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

class LocalContrastNormalization(object):
    def __init__(self, kernel_size=9, epsilon=1e-6):
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img).astype(np.float32) / 255.0

        pil_blur = img.filter(ImageFilter.GaussianBlur(radius=self.kernel_size // 2))
        blur_np = np.array(pil_blur).astype(np.float32) / 255.0

        img_sq = img_np * img_np
        pil_blur_sq = img.filter(ImageFilter.GaussianBlur(radius=self.kernel_size // 2))
        blur_sq_np = np.array(pil_blur_sq).astype(np.float32) / 255.0

        sigma = np.sqrt(np.maximum(blur_sq_np - blur_np ** 2, self.epsilon))

        out = np.zeros_like(img_np)
        for c in range(3):
            out[..., c] = (img_np[..., c] - blur_np[..., c]) / (sigma[..., c] + self.epsilon)

        out_min, out_max = out.min(), out.max()
        out = (out - out_min) / (out_max - out_min + self.epsilon)
        out = (out * 255.0).astype(np.uint8)
        return Image.fromarray(out)

def prepare_camvid_dataset():
    dataset_path = "CamVid"
    os.makedirs(dataset_path, exist_ok=True)
    data_dir = os.path.join(dataset_path, "data")

    if not os.path.exists(data_dir):
        print("Downloading CamVid dataset...")
        url = "https://github.com/alexgkendall/SegNet-Tutorial/archive/refs/heads/master.zip"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open("camvid_temp.zip", 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Extracting dataset...")
        with zipfile.ZipFile("camvid_temp.zip", 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        src_path = os.path.join(dataset_path, "SegNet-Tutorial-master", "CamVid")
        os.rename(src_path, data_dir)
        os.remove("camvid_temp.zip")
        shutil.rmtree(os.path.join(dataset_path, "SegNet-Tutorial-master"))
        print("Dataset preparation complete!")
    return data_dir

class CamVidDataset(Dataset):
    def __init__(self, root, split='train', image_size=(360, 480)):
        super().__init__()
        self.split = split
        self.root = root
        self.image_size = image_size

        list_file = os.path.join(root, f"{split}.txt")
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Не найден файл: {list_file}")

        with open(list_file, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        self.pairs = []
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

        self.lcn = LocalContrastNormalization(kernel_size=9)
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
    class_ious = torch.zeros(num_classes, device=preds.device)
    for cls in range(num_classes):
        pred_cls = (pred_labels == cls)
        true_cls = (labels == cls)
        intersection = torch.logical_and(pred_cls, true_cls).sum().float()
        union = torch.logical_or(pred_cls, true_cls).sum().float()
        if union > 0:
            class_ious[cls] = intersection / union
        else:
            class_ious[cls] = float('nan')
    mean_iou = float(torch.nanmean(class_ious))
    return mean_iou, class_ious
    
def calculate_class_accuracy(preds, labels, num_classes):
    pred_labels = preds.argmax(dim=1)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for cls in range(num_classes):
        cls_mask = (labels == cls)
        class_total[cls] = cls_mask.sum().item()
        if class_total[cls] > 0:
            class_correct[cls] = (pred_labels[cls_mask] == cls).sum().item()

    class_acc = class_correct / (class_total + 1e-8)
    return class_acc.mean().item(), class_acc

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

class SegNet(nn.Module):
    def __init__(self, num_classes=11):
        super(SegNet, self).__init__()

        self.encoder = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        ])

        self.decoder = nn.ModuleList([
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1, bias=False)
        ])
        self.pool_indices = []
        self.pool_output_sizes = []

    def forward(self, x):
        self.pool_indices = []
        self.pool_output_sizes = []

        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                self.pool_output_sizes.append(x.size())
                x, indices = layer(x)
                self.pool_indices.append(indices)
            else:
                x = layer(x)

        pool_idx = len(self.pool_indices) - 1
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                indices = self.pool_indices[pool_idx]
                out_size = self.pool_output_sizes[pool_idx]
                x = layer(x, indices, output_size=out_size)
                pool_idx -= 1
            else:
                x = layer(x)

        return x

def train_segnet_model(model, train_loader, val_loader, num_epochs, num_classes, lr, momentum, device):
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_val_acc = 0.0
    best_state  = None

    train_history = {'loss': [], 'global_acc': [], 'class_acc': [], 'miou': []}
    val_history   = {'global_acc': [], 'class_acc': [], 'miou': []}
    start_time = time.time()
    memory_usage = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_global_acc = 0.0
        epoch_class_acc = 0.0
        epoch_miou = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                global_acc = calculate_global_accuracy(outputs, labels)
                class_acc, _ = calculate_class_accuracy(outputs, labels, num_classes)
                miou, _ = calculate_miou(outputs, labels, num_classes)

            epoch_loss += loss.item()
            epoch_global_acc += global_acc
            epoch_class_acc += class_acc
            epoch_miou += miou

            memory_usage.append(psutil.virtual_memory().used / (1024 ** 3))

        num_batches = len(train_loader)
        train_history['loss'].append(epoch_loss / num_batches)
        train_history['global_acc'].append(epoch_global_acc / num_batches)
        train_history['class_acc'].append(epoch_class_acc / num_batches)
        train_history['miou'].append(epoch_miou / num_batches)

        model.eval()
        val_global_acc = 0.0
        val_class_acc = 0.0
        val_miou = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                val_global_acc += calculate_global_accuracy(outputs, labels)
                class_acc, _ = calculate_class_accuracy(outputs, labels, num_classes)
                miou, _ = calculate_miou(outputs, labels, num_classes)

                val_class_acc += class_acc
                val_miou += miou

        num_batches = len(val_loader)
        avg_val_acc = val_global_acc / num_batches
        val_history['global_acc'].append(avg_val_acc)
        val_history['class_acc'].append(val_class_acc / num_batches)
        val_history['miou'].append(val_miou / num_batches)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_history['loss'][-1]:.4f} | Global Acc: {train_history['global_acc'][-1]:.4f} | mIoU: {train_history['miou'][-1]:.4f}")
        print(f"Val Global Acc: {val_history['global_acc'][-1]:.4f} | mIoU: {val_history['miou'][-1]:.4f}")

    total_time = time.time() - start_time
    avg_memory = np.mean(memory_usage)

    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Average memory usage: {avg_memory:.2f} GB")

    model.load_state_dict(best_state)
    
    return train_history, val_history, total_time, avg_memory

def test_segnet_model(model, test_loader):
    model.eval()
    test_global_acc = 0.0
    test_class_acc = 0.0
    test_miou = 0.0
    class_ious = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            test_global_acc += calculate_global_accuracy(outputs, labels)
            class_acc, _ = calculate_class_accuracy(outputs, labels, num_classes)
            miou, ious = calculate_miou(outputs, labels, num_classes)

            test_class_acc += class_acc
            test_miou += miou
            class_ious += ious

    num_batches = len(test_loader)
    test_global_acc /= num_batches
    test_class_acc /= num_batches
    test_miou /= num_batches
    class_ious /= num_batches

    print(f"Test Results:")
    print(f"Global Accuracy: {test_global_acc:.4f}")
    print(f"Class Average Accuracy: {test_class_acc:.4f}")
    print(f"mIoU: {test_miou:.4f}")

    return test_global_acc, test_class_acc, test_miou, class_ious

image_size = (360, 480)
batch_size = 12
num_epochs = 100
num_classes = 11
lr = 0.1
momentum = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_path = prepare_camvid_dataset()

train_dataset = CamVidDataset(dataset_path, split='train', image_size=image_size)
val_dataset   = CamVidDataset(dataset_path, split='val',   image_size=image_size)
test_dataset  = CamVidDataset(dataset_path, split='test',  image_size=image_size)

print(f"Train: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Counting pixel frequencies (for median frequency balancing)")
pixel_counts = np.zeros(num_classes, dtype=np.float64)
total_pixels = 0

for images, labels in train_loader:
    lbl_np = labels.cpu().numpy().reshape(-1)
    for cls in range(num_classes):
        pixel_counts[cls] += np.sum(lbl_np == cls)
    total_pixels += np.sum(lbl_np < num_classes)

freqs = pixel_counts / total_pixels
median_freq = np.median(freqs)
class_weights = median_freq / freqs
class_weights = torch.from_numpy(class_weights.astype(np.float32)).to(device)
print("Class frequencies:", freqs)
print("Class weights (median freq):", class_weights.cpu().numpy())

model = SegNet(num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params/1e6:.2f}M")
print(f"Trainable Parameters: {trainable_params/1e6:.2f}M")
train_hist, val_hist, train_time, avg_mem = train_segnet_model(model, train_loader, val_loader,
                                                                num_epochs=num_epochs, num_classes=num_classes,
                                                                lr=lr, momentum=momentum, device=device
                                                              )
test_global_acc, test_class_acc, test_miou, class_ious = test_segnet_model(model, test_loader)

print(f'total_params: {total_params}',
    f'train_time: {train_time}'
    f'avg_memory: {avg_mem}',
    f'test_global_acc: {test_global_acc}',
    f'test_class_acc: {test_class_acc}',
    f'test_miou: {test_miou}',
    f'class_ious: {class_ious.cpu().numpy()}')

visualize_results(train_hist, val_hist, class_ious)