# Encoder-decoder vs Transformers

## Использованные статьи

- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://ieeexplore.ieee.org/document/7803544)
- [SegFormer: simple and efficient design for semantic segmentation with transformers](https://dl.acm.org/doi/10.5555/3540261.3541185)

### 1. Сегментация с использованием SegNet

Полученные результаты и графики:

### 2. Сегментация с использованием трансформеров ( SegformerForSemanticSegmentation)

В качестве трансформера взят SegformerForSemanticSegmentation (MiT B0 в роли энкодера внутри модели, All-MLP Head в роли декодера)

Модель была предобучена сначала на ImageNet, затем на ADE20K семантической сегментации (за счёт загрузки весов segformer-b0-finetuned-ade-512-512)

Итоговые результаты после 100 эпох обучения:  
Train: Loss: 0.3920, Acc: 0.8660, mIoU: 0.4440  
Val: Loss: 0.2842, Acc: 0.9001, mIoU: 0.5372  
Best Val mIoU: 0.5383  

Training completed in 29.10 minutes  
Average memory usage: 3.13 GB  


Результаты тестирования: Acc: 0.8634, mIoU: 0.4534

![alt text](<Снимок экрана 2025-06-06 в 23.49.39.png>)