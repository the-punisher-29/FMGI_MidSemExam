# SED Model Usage Guide

This repository contains a semantic segmentation model implementation with demo scripts, configuration files, and utilities for training and inference.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Demo Scripts](#demo-scripts)
- [Configuration](#configuration)
- [Model Inference](#model-inference)
- [Ground Truth Visualization](#ground-truth-visualization)
- [Dataset Preparation](#dataset-preparation)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)

## Project Structure

```
Paper/
├── configs/                    # Configuration files
│   ├── config.yaml            # Base configuration
│   ├── convnextB_768.yaml     # ConvNeXt-B model config
│   └── convnextL_768.yaml     # ConvNeXt-L model config
├── demo/                       # Demo scripts
│   ├── demo_for_vis.py        # Model inference visualization
│   ├── demo_for_gt.py         # Ground truth visualization
│   ├── predictor.py           # Prediction utilities
│   └── visualizer.py          # Visualization utilities
├── sed/                        # Core model implementation
│   ├── config.py              # Configuration management
│   ├── sed_model.py           # Main model architecture
│   └── test_time_augmentation.py
├── new_dataset/                # Dataset utilities
│   ├── conversion.py          # Data conversion scripts
│   ├── label_colors.txt       # Class color mappings
│   ├── cgt/                   # Color ground truth masks
│   ├── gt/                    # Ground truth masks
│   ├── input/                 # Input images
│   └── output/                # Model outputs
├── test_images/                # Test image samples
├── output_images/              # Generated outputs
│   ├── sed_model_base/        # Base model outputs
│   └── sed_model_large/       # Large model outputs
├── miou_val.py                # Mean IoU evaluation
└── requirements.txt           # Python dependencies
```

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13 is recommended and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

An example of installation is shown below:

```bash
git clone https://github.com/the-punisher-29/FMGI_MidSemExam.git
cd FMGI_MidSemExam/Paper
conda create -n sed python=3.8
conda activate sed
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
cd open_clip/
pip install -e .
cd ..
```
## Demo Scripts

### 1. Model Inference Visualization (`demo_for_vis.py`)

**Primary Purpose:** Runs model inference on images and generates prediction visualizations.

**Basic Usage:**
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "test_images/*" \
    --output output_images/sed_model_base/ \
    --opts MODEL.WEIGHTS models/sed_model_base.pth
```

**Advanced Usage with Custom Paths:**
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextL_768.yaml \
    --input "E:/Pr/Course_IITJ/FMGI/Minor_Exam/SED/test_images/*" \
    --output output_images/sed_model_large/ \
    --opts MODEL.WEIGHTS models/sed_model_large.pth \
           INPUT.MIN_SIZE_TEST 768 \
           INPUT.MAX_SIZE_TEST 768
```

**Parameters:**
- `--config-file`: Model configuration file
- `--input`: Input images path (supports wildcards)
- `--output`: Output directory for visualizations
- `--opts`: Override configuration options
  - `MODEL.WEIGHTS`: Path to trained model weights
  - `INPUT.MIN_SIZE_TEST`: Minimum test image size
  - `INPUT.MAX_SIZE_TEST`: Maximum test image size

### 2. Ground Truth Visualization (`demo_for_gt.py`)

**Primary Purpose:** Visualizes ground truth segmentation masks without model inference.

**Basic Usage:**
```bash
python demo/demo_for_gt.py \
    --config-file configs/convnextB_768.yaml \
    --input "new_dataset/input/*.jpg" \
    --gt "new_dataset/gt/" \
    --output "output_images/ground_truth_vis/"
```

**Parameters:**
- `--config-file`: Configuration file for visualization settings
- `--input`: Input images path
- `--gt`: Ground truth masks directory
- `--output`: Output directory for GT visualizations

## Configuration

### Model Configurations

1. **ConvNeXt-Base (convnextB_768.yaml):**
   - Input resolution: 768x768
   - Model: ConvNeXt-Base backbone
   - Suitable for: Standard inference, good balance of speed and accuracy

2. **ConvNeXt-Large (convnextL_768.yaml):**
   - Input resolution: 768x768
   - Model: ConvNeXt-Large backbone
   - Suitable for: High-accuracy inference, slower but more precise

### Custom Configuration
Edit `configs/config.yaml` or create a new config file:
```yaml
MODEL:
  WEIGHTS: "path/to/your/model.pth"
  DEVICE: "cuda"  # or "cpu"
  
INPUT:
  MIN_SIZE_TEST: 768
  MAX_SIZE_TEST: 768
  
OUTPUT:
  VIS_THRESH: 0.5
```

## Model Inference

### Single Image Inference
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "path/to/single_image.jpg" \
    --output "output/" \
    --opts MODEL.WEIGHTS "models/sed_model_base.pth"
```

### Batch Image Inference
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "input_folder/*.jpg" \
    --output "output_folder/" \
    --opts MODEL.WEIGHTS "models/sed_model_base.pth"
```

### Multiple Image Formats
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "input_folder/*.{jpg,png,jpeg}" \
    --output "output_folder/" \
    --opts MODEL.WEIGHTS "models/sed_model_base.pth"
```

## Ground Truth Visualization

### Basic GT Visualization
```bash
python demo/demo_for_gt.py \
    --config-file configs/convnextB_768.yaml \
    --input "new_dataset/input/*.jpg" \
    --gt "new_dataset/cgt/" \
    --output "gt_visualizations/"
```

### Compare Input and GT Side-by-side
```bash
python demo/demo_for_gt.py \
    --config-file configs/convnextB_768.yaml \
    --input "new_dataset/input/*.jpg" \
    --gt "new_dataset/cgt/" \
    --output "comparison_output/" \
    --opts VISUALIZE.SIDE_BY_SIDE True
```

## Dataset Preparation

### Convert RGB to Greyscale Groundtruth
```bash
python new_dataset/conversion.py 
```

### Label Color Mapping
Edit `new_dataset/label_colors.txt` to define class colors:
```
0 0 0 0        # Background
1 128 0 0      # Class 1 - Red
2 0 128 0      # Class 2 - Green
3 0 0 128      # Class 3 - Blue
...
```

## Evaluation

### Calculate Mean IoU
```bash
python miou_val.py \
    --pred-dir "output_images/sed_model_base/" \
    --gt-dir "new_dataset/gt/" \
    --num-classes xx
```

### Evaluation with Specific Classes
```bash
python miou_val.py \
    --pred-dir "output_images/sed_model_base/" \
    --gt-dir "new_dataset/gt/" \
    --num-classes 21 \
    --ignore-classes 0,255  # Ignore background and void classes
```

## Advanced Usage

### Custom Model Weights
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "test_images/*" \
    --output "custom_output/" \
    --opts MODEL.WEIGHTS "path/to/custom_weights.pth" \
           MODEL.DEVICE "cuda:1"
```

### Test Time Augmentation
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "test_images/*" \
    --output "tta_output/" \
    --opts MODEL.WEIGHTS "models/sed_model_base.pth" \
           TEST.AUG.ENABLED True \
           TEST.AUG.SCALES "[0.8, 1.0, 1.2]" \
           TEST.AUG.FLIP True
```

### Multi-GPU Inference
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextL_768.yaml \
    --input "large_dataset/*" \
    --output "multi_gpu_output/" \
    --opts MODEL.WEIGHTS "models/sed_model_large.pth" \
           MODEL.DEVICE "cuda:0" \
           DATALOADER.NUM_WORKERS 8
```

### Memory-Efficient Inference
```bash
python demo/demo_for_vis.py \
    --config-file configs/convnextB_768.yaml \
    --input "test_images/*" \
    --output "memory_efficient_output/" \
    --opts MODEL.WEIGHTS "models/sed_model_base.pth" \
           INPUT.MIN_SIZE_TEST 512 \
           INPUT.MAX_SIZE_TEST 512 \
           DATALOADER.NUM_WORKERS 2
```

## Common Issues and Solutions

### 1. CUDA Out of Memory
- Reduce batch size or image resolution
- Use CPU inference: `--opts MODEL.DEVICE "cpu"`

### 2. Model Weights Not Found
- Check the path to model weights
- Download pre-trained weights if needed

### 3. Input Format Issues
- Ensure input images are in supported formats (jpg, png, jpeg)
- Check file permissions and paths

### 4. Visualization Issues
- Verify output directory exists and is writable
- Check if visualization dependencies are installed

## Output Formats

### Prediction Outputs
- **Raw predictions:** `.npy` files with class probabilities
- **Visualizations:** `.png` files with colored segmentation masks
- **Overlays:** Original image with transparent mask overlay

### Ground Truth Outputs
- **Color-coded masks:** Using colors from `label_colors.txt`
- **Side-by-side comparisons:** Input image and GT mask
- **Overlay visualizations:** GT mask overlaid on input image

## Performance Tips

1. **Use appropriate image sizes** based on your GPU memory
2. **Enable mixed precision** for faster inference
3. **Use multiple workers** for data loading
4. **Consider test-time augmentation** for better accuracy
5. **Use batch processing** for multiple images

## Support

For issues and questions:
- Check the configuration files for proper settings
- Ensure all dependencies are installed correctly
- Verify model weights are compatible with the configuration
- Review the error logs for specific issues

---