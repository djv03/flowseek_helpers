#!/usr/bin/env python3
"""
Simple FlowSeek Test Script
Tests if FlowSeek works on a single image pair
"""

import sys
import os
from pathlib import Path

# Suppress matplotlib warning about disk quota
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'

import numpy as np
import torch
from PIL import Image

# FlowSeek root
FLOWSEEK_ROOT = Path("/proj/ciptmp/we03cyna/flowseek_cloned")
sys.path.insert(0, str(FLOWSEEK_ROOT))

print("=" * 80)
print("FlowSeek Single Pair Test")
print("=" * 80)

# Check command line args
if len(sys.argv) != 4:
    print("\nUsage: python test_flowseek.py <img1> <img2> <output.flo>")
    print("\nExample:")
    print("  python test_flowseek.py \\")
    print("    /path/to/frame_0001.png \\")
    print("    /path/to/frame_0002.png \\")
    print("    test_flow.flo")
    sys.exit(1)

img1_path = sys.argv[1]
img2_path = sys.argv[2]
output_path = sys.argv[3]

print(f"\nInput images:")
print(f"  Image 1: {img1_path}")
print(f"  Image 2: {img2_path}")
print(f"  Output:  {output_path}")

# Check files exist
if not Path(img1_path).exists():
    print(f"\nERROR: Image 1 not found: {img1_path}")
    sys.exit(1)

if not Path(img2_path).exists():
    print(f"\nERROR: Image 2 not found: {img2_path}")
    sys.exit(1)

# Model paths
config_path = FLOWSEEK_ROOT / "config/eval/flowseek-T.json"
weights_dir = FLOWSEEK_ROOT / "weights"

# Find a weight file
weight_files = list(weights_dir.glob("flowseek*.pth"))
if not weight_files:
    print(f"\nERROR: No FlowSeek weights found in {weights_dir}")
    print("Run: cd /proj/ciptmp/we03cyna/flowseek_cloned && bash scripts/get_weights.sh")
    sys.exit(1)

model_path = weight_files[0]

print(f"\nFlowSeek setup:")
print(f"  Config: {config_path}")
print(f"  Model:  {model_path}")

# Check if files exist
if not config_path.exists():
    print(f"\nERROR: Config not found: {config_path}")
    sys.exit(1)

if not model_path.exists():
    print(f"\nERROR: Model not found: {model_path}")
    sys.exit(1)

print("\n" + "=" * 80)
print("Loading FlowSeek...")
print("=" * 80)

try:
    # Import FlowSeek - it's in core/flowseek.py
    from core.flowseek import FlowSeek
    print("✓ Imported core.flowseek.FlowSeek")
    
    import json
    
    # Load config
    print(f"✓ Loading config from {config_path.name}")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Initialize model
    print("✓ Initializing FlowSeek model...")
    model = FlowSeek(cfg)
    
    # Load checkpoint
    print(f"✓ Loading weights from {model_path.name}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        print("✓ Moving model to GPU")
        model.cuda()
        device = "GPU"
    else:
        print("✓ Using CPU (this will be slow)")
        device = "CPU"
    
    print(f"\n{'=' * 80}")
    print(f"Running inference on {device}...")
    print("=" * 80)
    
    # Load images
    def load_image(imfile):
        img = Image.open(imfile).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img
    
    print("✓ Loading images...")
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    print(f"  Image shape: {img1.shape}")
    
    # Add batch dimension
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    
    # Run inference
    print("✓ Running FlowSeek inference (this may take a moment)...")
    with torch.no_grad():
        flow_predictions = model(img1, img2, iters=32)
        flow = flow_predictions[-1][0].cpu().numpy()
    
    # Convert from (2, H, W) to (H, W, 2)
    flow = flow.transpose(1, 2, 0)
    
    print(f"✓ Flow computed! Shape: {flow.shape}")
    print(f"  Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
    print(f"  Flow magnitude mean: {np.sqrt(np.sum(flow**2, axis=2)).mean():.2f} pixels")
    
    # Save flow
    print(f"✓ Saving flow to: {output_path}")
    
    def write_flow(filename, uv):
        """Write .flo file"""
        with open(filename, 'wb') as f:
            np.array([202021.25], dtype=np.float32).tofile(f)
            np.array([uv.shape[1], uv.shape[0]], dtype=np.int32).tofile(f)
            uv.astype(np.float32).tofile(f)
    
    write_flow(output_path, flow)
    
    print("\n" + "=" * 80)
    print("SUCCESS! FlowSeek is working correctly.")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")
    print("You can now run the full comparison pipeline!")
    
except ImportError as e:
    print(f"\n{'=' * 80}")
    print("ERROR: Failed to import FlowSeek")
    print("=" * 80)
    print(f"\nError message: {e}")
    print("\nTroubleshooting:")
    print("1. Install FlowSeek dependencies:")
    print(f"   cd {FLOWSEEK_ROOT}")
    print("   pip install -r requirements.txt")
    print("\n2. Check that FlowSeek files exist:")
    print(f"   ls {FLOWSEEK_ROOT}/core/")
    sys.exit(1)

except Exception as e:
    print(f"\n{'=' * 80}")
    print("ERROR: FlowSeek inference failed")
    print("=" * 80)
    print(f"\nError message: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)