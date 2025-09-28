import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_colors_and_create_mapping(gt_folder):
    """
    Extract all unique colors from ground truth images and create automatic mapping
    """
    print("Scanning ground truth images to extract unique colors...")
    
    gt_folder = Path(gt_folder)
    gt_files = sorted([f for f in gt_folder.glob('*') 
                      if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    if not gt_files:
        raise ValueError(f"No ground truth images found in {gt_folder}")
    
    all_colors = set()
    
    # Extract all unique colors from all GT images
    for gt_file in gt_files[:10]:  # Sample first 10 images to get color palette
        try:
            img = cv2.imread(str(gt_file), cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
                for color in unique_colors:
                    all_colors.add(tuple(color))
        except Exception as e:
            print(f"Error reading {gt_file}: {e}")
    
    # Sort colors for consistent mapping
    sorted_colors = sorted(list(all_colors))
    
    # Create color to class mapping
    color_map = {}
    class_names = {}
    
    for class_id, (r, g, b) in enumerate(sorted_colors):
        color_map[(r, g, b)] = class_id
        # Create descriptive names based on color
        class_names[class_id] = f"class_{class_id}_rgb({r},{g},{b})"
    
    print(f"Found {len(sorted_colors)} unique colors:")
    for i, (r, g, b) in enumerate(sorted_colors[:10]):  # Show first 10
        print(f"  Class {i}: RGB({r:3d}, {g:3d}, {b:3d})")
    if len(sorted_colors) > 10:
        print(f"  ... and {len(sorted_colors) - 10} more colors")
    
    return color_map, class_names

def rgb_to_class_mask(rgb_image, color_map):
    """Convert RGB image to class indices using color mapping"""
    h, w = rgb_image.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int32)
    
    # Reshape for efficient processing
    rgb_reshaped = rgb_image.reshape(-1, 3)
    class_reshaped = np.zeros(rgb_reshaped.shape[0], dtype=np.int32)
    
    # Map each unique color to its class
    unique_colors, inverse_indices = np.unique(rgb_reshaped, axis=0, return_inverse=True)
    
    for i, color in enumerate(unique_colors):
        r, g, b = color
        if (r, g, b) in color_map:
            class_id = color_map[(r, g, b)]
        else:
            # Assign unknown colors to class 0
            class_id = 0
            print(f"Warning: Unknown color RGB({r}, {g}, {b}) assigned to class 0")
        
        # Update all pixels with this color
        mask = (inverse_indices == i)
        class_reshaped[mask] = class_id
    
    return class_reshaped.reshape(h, w)

def calculate_iou(pred_mask, gt_mask, num_classes):
    """Calculate IoU for each class"""
    iou_per_class = []
    
    for class_id in range(num_classes):
        # Binary masks for current class
        pred_binary = (pred_mask == class_id)
        gt_binary = (gt_mask == class_id)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Calculate IoU
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    return np.array(iou_per_class)

def calculate_pixel_accuracy(pred_mask, gt_mask):
    """Calculate pixel accuracy"""
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = pred_mask.size
    return correct_pixels / total_pixels

def load_image_rgb(image_path):
    """Load image and convert to RGB"""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def evaluate_segmentation_dataset(pred_folder, gt_folder):
    """
    Evaluate segmentation dataset with automatic color mapping
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)
    
    print("=" * 60)
    print("AUTOMATIC RGB SEGMENTATION EVALUATION")
    print("=" * 60)
    
    # Step 1: Extract color mapping from ground truth images
    color_map, class_names = extract_colors_and_create_mapping(gt_folder)
    num_classes = len(color_map)
    
    # Step 2: Get prediction files
    pred_files = sorted([f for f in pred_folder.glob('*') 
                        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    if not pred_files:
        raise ValueError(f"No prediction images found in {pred_folder}")
    
    print(f"\nProcessing {len(pred_files)} image pairs...")
    
    # Step 3: Process each image pair
    all_ious = []
    all_pixel_accuracies = []
    per_image_results = []
    processed_count = 0
    
    for pred_file in pred_files:
        # Find corresponding ground truth
        base_name = pred_file.stem
        possible_gt_names = [
            f"{base_name}_L{pred_file.suffix}",
            f"{base_name}_L.png",
            f"{base_name}_L.jpg",
            f"{base_name}_gt{pred_file.suffix}",
            f"{base_name}_gt.png"
        ]
        
        gt_file = None
        for gt_name in possible_gt_names:
            potential_gt = gt_folder / gt_name
            if potential_gt.exists():
                gt_file = potential_gt
                break
        
        if gt_file is None:
            print(f"Warning: No ground truth found for {pred_file.name}")
            continue
        
        try:
            # Load images
            pred_rgb = load_image_rgb(pred_file)
            gt_rgb = load_image_rgb(gt_file)
            
            # Resize if dimensions don't match
            if pred_rgb.shape != gt_rgb.shape:
                print(f"Resizing {pred_file.name}: {pred_rgb.shape} -> {gt_rgb.shape}")
                pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Convert RGB to class masks
            pred_mask = rgb_to_class_mask(pred_rgb, color_map)
            gt_mask = rgb_to_class_mask(gt_rgb, color_map)
            
            # Calculate metrics
            iou_per_class = calculate_iou(pred_mask, gt_mask, num_classes)
            pixel_accuracy = calculate_pixel_accuracy(pred_mask, gt_mask)
            mean_iou = np.mean(iou_per_class)
            
            # Store results
            all_ious.append(iou_per_class)
            all_pixel_accuracies.append(pixel_accuracy)
            
            per_image_results.append({
                'image': pred_file.name,
                'mean_iou': mean_iou,
                'pixel_accuracy': pixel_accuracy,
                'class_ious': iou_per_class
            })
            
            processed_count += 1
            print(f"✓ {pred_file.name}: mIoU={mean_iou:.3f}, PA={pixel_accuracy:.3f}")
            
        except Exception as e:
            print(f"✗ Error processing {pred_file.name}: {e}")
            continue
    
    if processed_count == 0:
        raise ValueError("No valid image pairs processed!")
    
    # Step 4: Calculate overall metrics
    all_ious = np.array(all_ious)
    class_mean_ious = np.mean(all_ious, axis=0)
    overall_mean_iou = np.mean(class_mean_ious)
    overall_pixel_accuracy = np.mean(all_pixel_accuracies)
    
    results = {
        'overall_mean_iou': overall_mean_iou,
        'overall_pixel_accuracy': overall_pixel_accuracy,
        'class_mean_ious': class_mean_ious,
        'num_classes': num_classes,
        'num_images': processed_count,
        'per_image_results': per_image_results,
        'color_map': color_map,
        'class_names': class_names
    }
    
    return results

def plot_results(results, save_path='segmentation_results.png'):
    """Create visualization of results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    class_ious = results['class_mean_ious']
    per_image_results = results['per_image_results']
    class_names = results['class_names']
    
    # 1. Per-class IoU
    ax = axes[0, 0]
    x_pos = range(len(class_ious))
    bars = ax.bar(x_pos, class_ious, alpha=0.7)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU per Class')
    ax.grid(True, alpha=0.3)
    
    # Color bars based on performance
    for i, (bar, iou) in enumerate(zip(bars, class_ious)):
        if iou > 0.7:
            bar.set_color('green')
        elif iou > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 2. Per-image mIoU
    ax = axes[0, 1]
    image_ious = [item['mean_iou'] for item in per_image_results]
    ax.plot(range(len(image_ious)), image_ious, 'o-', markersize=3)
    ax.axhline(y=results['overall_mean_iou'], color='r', linestyle='--', 
               label=f"Overall mIoU: {results['overall_mean_iou']:.3f}")
    ax.set_xlabel('Image Index')
    ax.set_ylabel('mIoU')
    ax.set_title('mIoU per Image')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Pixel accuracy per image
    ax = axes[1, 0]
    image_pas = [item['pixel_accuracy'] for item in per_image_results]
    ax.plot(range(len(image_pas)), image_pas, 'o-', markersize=3, color='green')
    ax.axhline(y=results['overall_pixel_accuracy'], color='r', linestyle='--',
               label=f"Overall PA: {results['overall_pixel_accuracy']:.3f}")
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Pixel Accuracy')
    ax.set_title('Pixel Accuracy per Image')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. mIoU vs Pixel Accuracy scatter
    ax = axes[1, 1]
    ax.scatter(image_pas, image_ious, alpha=0.6)
    ax.set_xlabel('Pixel Accuracy')
    ax.set_ylabel('mIoU')
    ax.set_title('mIoU vs Pixel Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line
    min_val = min(min(image_pas), min(image_ious))
    max_val = max(max(image_pas), max(image_ious))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_color_mapping(color_map, class_names, filename='extracted_color_map.txt'):
    """Save the extracted color mapping for future use"""
    with open(filename, 'w') as f:
        f.write("# Automatically extracted color mapping\n")
        f.write("# Format: class_id r g b class_name\n")
        for (r, g, b), class_id in sorted(color_map.items(), key=lambda x: x[1]):
            class_name = class_names.get(class_id, f'class_{class_id}')
            f.write(f"{class_id} {r} {g} {b} {class_name}\n")
    print(f"Color mapping saved to: {filename}")

def main():
    # Configuration - UPDATE THESE PATHS
    pred_folder = "new_dataset/output/l"    # Your predictions folder
    gt_folder = "new_dataset/gt"   # Your ground truth folder
    
    print("Direct RGB Segmentation Evaluation")
    print("(No color mapping file required)")
    print(f"Predictions: {pred_folder}")
    print(f"Ground Truth: {gt_folder}")
    
    try:
        # Evaluate dataset
        results = evaluate_segmentation_dataset(pred_folder, gt_folder)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Mean IoU:      {results['overall_mean_iou']:.4f}")
        print(f"Overall Pixel Accuracy: {results['overall_pixel_accuracy']:.4f}")
        print(f"Number of Classes:     {results['num_classes']}")
        print(f"Images Processed:      {results['num_images']}")
        
        print(f"\nTop 10 Class Performance:")
        class_ious = results['class_mean_ious']
        sorted_classes = sorted(enumerate(class_ious), key=lambda x: x[1], reverse=True)
        
        for i, (class_id, iou) in enumerate(sorted_classes[:10]):
            print(f"  {i+1:2d}. Class {class_id:2d}: {iou:.4f}")
        
        print(f"\nWorst 5 Class Performance:")
        for i, (class_id, iou) in enumerate(sorted_classes[-5:]):
            print(f"     Class {class_id:2d}: {iou:.4f}")
        
        # Find best and worst images
        image_ious = [item['mean_iou'] for item in results['per_image_results']]
        best_idx = np.argmax(image_ious)
        worst_idx = np.argmin(image_ious)
        
        best_result = results['per_image_results'][best_idx]
        worst_result = results['per_image_results'][worst_idx]
        
        print(f"\nBest Image:  {best_result['image']} (mIoU: {best_result['mean_iou']:.4f})")
        print(f"Worst Image: {worst_result['image']} (mIoU: {worst_result['mean_iou']:.4f})")
        
        # Save detailed results
        with open('evaluation_results.txt', 'w') as f:
            f.write("RGB Segmentation Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Mean IoU:       {results['overall_mean_iou']:.6f}\n")
            f.write(f"Overall Pixel Accuracy: {results['overall_pixel_accuracy']:.6f}\n")
            f.write(f"Number of Classes:      {results['num_classes']}\n")
            f.write(f"Images Processed:       {results['num_images']}\n\n")
            
            f.write("Per-Class IoU Results:\n")
            for class_id, iou in enumerate(results['class_mean_ious']):
                f.write(f"Class {class_id:2d}: {iou:.6f}\n")
            
            f.write(f"\nPer-Image Results:\n")
            for item in results['per_image_results']:
                f.write(f"{item['image']:30s}: mIoU={item['mean_iou']:.4f}, "
                       f"PA={item['pixel_accuracy']:.4f}\n")
        
        # Save extracted color mapping for future use
        save_color_mapping(results['color_map'], results['class_names'])
        
        # Create visualization
        plot_results(results)
        
        print(f"\n✓ Detailed results saved to 'evaluation_results.txt'")
        print(f"✓ Color mapping saved to 'extracted_color_map.txt'")
        print(f"✓ Visualization saved to 'segmentation_results.png'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()