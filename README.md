# Image Style Transfer (Neural Style Transfer)

This is a comprehensive **Image Style Transfer** project implementing two approaches:

1. **Optimization-based (Gatys et al.)** - Original slow method using pixel optimization
2. **Feedforward AdaIN (Huang & Wang, CVPR 2017)** - Fast arbitrary style transfer

---

## Project Structure

```
Image Style Transfer/
├── models/
│   ├── __init__.py
│   └── adain_model.py       # AdaIN architecture (encoder-decoder)
├── datasets/
│   ├── __init__.py
│   └── datasets.py          # Content & Style dataset loaders
├── losses/
│   ├── __init__.py
│   └── losses.py            # Perceptual losses & evaluation metrics
├── train.py                 # Training script
├── test.py                  # Testing script (with metrics)
├── download_datasets.py     # Dataset setup helper
├── neural_style.py          # Original optimization-based method
├── stylize.py
├── vgg.py
├── saliency.py
├── segmentation.py
└── checkpoints/            # Trained models will be saved here
    └── final_model.pth
```

---

## Mode 2: AdaIN Arbitrary Style Transfer (Recommended)

**Key Feature:** At test time, you provide **BOTH content + style images** to generate a stylized output.

### Step 1: Setup Datasets

```powershell
# Create dataset folders
mkdir datasets/content
mkdir datasets/style
```

**Content Dataset Options:**

1. **MSCOCO (Recommended - 80K images)**
   - Download from: https://cocodataset.org/#download
   - Use '2017 Train images [123K/5GB]'
   - Extract to `datasets/content/`

2. **Small test set:**
   - Add any images you want to `datasets/content/`

**Style Dataset:**
- Add artistic images to `datasets/style/`
- Examples: paintings, artwork, patterns

### Step 2: Install Dependencies

```powershell
uv sync
```

### Step 3: Training

```powershell
# Basic training
uv run python train.py --content-dir ./datasets/content --style-dir ./datasets/style --output-dir ./checkpoints --epochs 2 --batch-size 8

# Custom training parameters
uv run python train.py `
  --content-dir ./datasets/content `
  --style-dir ./datasets/style `
  --output-dir ./checkpoints `
  --epochs 5 `
  --batch-size 16 `
  --lr 1e-4 `
  --image-size 256
```

### Step 4: Testing (Content + Style → Output)

```powershell
# Test with metrics computation
uv run python test.py `
  --model ./checkpoints/final_model.pth `
  --content ./examples/1-content.jpg `
  --style ./examples/1-style.jpg `
  --output ./examples/adain_output.jpg

# Test without metrics (faster)
uv run python test.py `
  --model ./checkpoints/final_model.pth `
  --content ./examples/1-content.jpg `
  --style ./examples/1-style.jpg `
  --output ./examples/adain_output.jpg `
  --no-metrics
```

---

## Mode 1: Original Optimization-based (Slow but Quality)

```powershell
uv run python neural_style.py --content .\examples\1-content.jpg --styles .\examples\1-style.jpg --output .\examples\out.jpg --iterations 1000 --overwrite
```

---

## Evaluation Metrics

The test script computes:

| Metric | Description | Range |
|--------|-------------|-------|
| **Style Similarity Score** | How well output matches style statistics (Gram matrix) | 0-1 |
| **Content SSIM** | Structural similarity between content and output | 0-1 |

**Interpretation:**
- Higher Style Similarity = Better style transfer
- Higher Content SSIM = Better content preservation
- Trade-off: Extreme style matching may reduce content preservation

---

## Architecture Overview

```
TRAINING:
┌──────────────┐     ┌──────────┐     ┌─────────────┐
│   Content    │     │  VGG-19  │────▶│   Decoder   │
│   Dataset     │────▶│ (frozen) │     │             │
└──────────────┘     └──────────┘     └─────────────┘
       │                                        │
       │                                        │ (backprop)
       ▼                                        ▼
┌──────────────┐                         ┌─────────────┐
│    Style     │                         │ Loss (Adam) │
│   Dataset    │                         └─────────────┘
└──────────────┘

TESTING:
┌────────────┐     ┌──────────────┐     ┌─────────────┐
│  Content   │     │  VGG-19 +    │     │             │
│   Image    │────▶│  AdaIN +     │────▶│   Output    │
│            │     │  Decoder     │     │  (Stylized) │
└────────────┘     └──────────────┘     └─────────────┘
       │
       ▼
┌────────────┐
│   Style    │
│   Image    │──── (applies style statistics)
└────────────┘
```

---

## Model Weights

Training saves:
- `./checkpoints/final_model.pth` - Complete trained model
- `./checkpoints/checkpoint_*.pth` - Intermediate checkpoints

---

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 2 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--image-size` | 256 | Training image size |
| `--content-weight` | 1.0 | Content loss weight |
| `--style-weight` | 1e2 | Style loss weight |
| `--tv-weight` | 1e-5 | Total variation loss weight |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- TensorFlow 2.19+
- CUDA-capable GPU (recommended for training)

---

## Training Time Estimates

| GPU | Image Size | Batch Size | Time/Epoch |
|-----|------------|------------|------------|
| RTX 3080+ | 256 | 8 | ~30 min |
| RTX 3080+ | 512 | 4 | ~1 hr |
| CPU | 256 | 4 | ~4+ hrs |

---

## Troubleshooting

**CUDA out of memory:**
```powershell
uv run python train.py --batch-size 4 --image-size 256
```

**No style images found:**
- Ensure images are in `./datasets/style/` with extensions `.jpg`, `.png`, `.jpeg`

**Model loading errors:**
- Check if checkpoint file exists: `./checkpoints/final_model.pth`

---

## License

GNU GPLv3