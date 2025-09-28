import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path

# CamVid color map based on your provided mapping
CAMVID_COLOR_MAP = {
    (64, 128, 64): 0,    # Animal
    (192, 0, 128): 1,    # Archway
    (0, 128, 192): 2,    # Bicyclist
    (0, 128, 64): 3,     # Bridge
    (128, 0, 0): 4,      # Building
    (64, 0, 128): 5,     # Car
    (64, 0, 192): 6,     # CartLuggagePram
    (192, 128, 64): 7,   # Child
    (192, 192, 128): 8,  # Column_Pole
    (64, 64, 128): 9,    # Fence
    (128, 0, 192): 10,   # LaneMkgsDriv
    (192, 0, 64): 11,    # LaneMkgsNonDriv
    (128, 128, 64): 12,  # Misc_Text
    (192, 0, 192): 13,   # MotorcycleScooter
    (128, 64, 64): 14,   # OtherMoving
    (64, 192, 128): 15,  # ParkingBlock
    (64, 64, 0): 16,     # Pedestrian
    (128, 64, 128): 17,  # Road
    (128, 128, 192): 18, # RoadShoulder
    (0, 0, 192): 19,     # Sidewalk
    (192, 128, 128): 20, # SignSymbol
    (128, 128, 128): 21, # Sky
    (64, 128, 192): 22,  # SUVPickupTruck
    (0, 0, 64): 23,      # TrafficCone
    (0, 64, 64): 24,     # TrafficLight
    (192, 64, 128): 25,  # Train
    (128, 128, 0): 26,   # Tree
    (192, 128, 192): 27, # Truck_Bus
    (64, 0, 64): 28,     # Tunnel
    (192, 192, 0): 29,   # VegetationMisc
    (0, 0, 0): 30,       # Void
    (64, 192, 0): 31     # Wall
}

# Class names for reference
CAMVID_CLASS_NAMES = [
    'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 
    'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
    'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
    'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk',
    'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight',
    'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'
]

def convert_rgb_to_class_indices(rgb_mask, color_map):
    """
    Convert RGB segmentation mask to class indices
    
    Args:
        rgb_mask: RGB image array (H, W, 3)
        color_map: Dictionary mapping RGB tuples to class indices
    
    Returns:
        class_mask: 2D array (H, W) with class indices
    """
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Track which colors we couldn't map
    unmapped_colors = set()
    
    for rgb_color, class_idx in color_map.items():
        # Find pixels matching this color (exact match)
        mask = np.all(rgb_mask == rgb_color, axis=2)
        class_mask[mask] = class_idx
    
    # Check for unmapped pixels
    mapped_pixels = np.zeros((h, w), dtype=bool)
    for rgb_color in color_map.keys():
        mask = np.all(rgb_mask == rgb_color, axis=2)
        mapped_pixels |= mask
    
    # Find unmapped colors
    unmapped_mask = ~mapped_pixels
    if np.any(unmapped_mask):
        unmapped_pixels = rgb_mask[unmapped_mask]
        for pixel in unmapped_pixels[:10]:  # Show first 10 unmapped colors
            unmapped_colors.add(tuple(pixel))
    
    if unmapped_colors:
        print(f"Warning: Found {len(unmapped_colors)} unmapped colors: {list(unmapped_colors)[:5]}...")
        # Set unmapped pixels to void class (30)
        class_mask[unmapped_mask] = 30
    
    return class_mask

def process_camvid_directory(rgb_dir, output_dir):
    """Process all CamVid ground truth files in a directory"""
    
    rgb_path = Path(rgb_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files
    rgb_files = list(rgb_path.glob("*.png"))
    
    if not rgb_files:
        print(f"No PNG files found in {rgb_dir}")
        return
    
    print(f"Found {len(rgb_files)} RGB ground truth files")
    
    for rgb_file in rgb_files:
        print(f"Processing {rgb_file.name}...")
        
        # Load RGB ground truth
        rgb_mask = np.array(Image.open(rgb_file))
        
        if len(rgb_mask.shape) != 3:
            print(f"Warning: {rgb_file.name} is not RGB, skipping...")
            continue
        
        # Convert to class indices
        class_mask = convert_rgb_to_class_indices(rgb_mask, CAMVID_COLOR_MAP)
        
        # Save converted mask
        output_file = output_path / rgb_file.name
        Image.fromarray(class_mask).save(output_file)
        
        print(f"  Saved to {output_file}")
        print(f"  Shape: {class_mask.shape}, Classes found: {len(np.unique(class_mask))}")
        print(f"  Class range: {class_mask.min()} to {class_mask.max()}")

def verify_conversion(class_mask_file, rgb_file=None):
    """Verify the conversion worked correctly"""
    class_mask = np.array(Image.open(class_mask_file))
    
    print(f"\nVerification for {class_mask_file}:")
    print(f"Shape: {class_mask.shape}")
    print(f"Data type: {class_mask.dtype}")
    print(f"Class range: {class_mask.min()} to {class_mask.max()}")
    
    unique_classes = np.unique(class_mask)
    print(f"Classes present: {unique_classes}")
    print("Class names present:")
    for class_id in unique_classes:
        if class_id < len(CAMVID_CLASS_NAMES):
            print(f"  {class_id}: {CAMVID_CLASS_NAMES[class_id]}")
        else:
            print(f"  {class_id}: Unknown class")

if __name__ == "__main__":
    # Example usage
    rgb_gt_dir = "gt"  # Update this path
    converted_gt_dir = "cgt"  # Update this path
    
    print("CamVid RGB to Class Index Converter")
    print("="*50)
    
    # Check if directories exist
    if not os.path.exists(rgb_gt_dir):
        print(f"RGB ground truth directory not found: {rgb_gt_dir}")
        print("Please update the rgb_gt_dir path in the script")
        exit(1)
    
    # Process directory
    process_camvid_directory(rgb_gt_dir, converted_gt_dir)
    
    # Verify a sample conversion
    converted_files = list(Path(converted_gt_dir).glob("*.png"))
    if converted_files:
        verify_conversion(converted_files[0])
    
    print(f"\nConversion complete! Converted files saved to: {converted_gt_dir}")
    print(f"You can now use these files with the demo script:")
    print(f"python demo/demo_for_gt.py --gt {converted_gt_dir}")