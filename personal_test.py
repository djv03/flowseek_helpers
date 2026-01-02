#!/usr/bin/env python3
"""
Process Personal Videos: FlowSeek vs Lucas-Kanade
Extracts frames and compares optical flow methods on real-world footage
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess

# Add FlowSeek to path
FLOWSEEK_ROOT = Path("/proj/ciptmp/we03cyna/flowseek_cloned")
sys.path.insert(0, str(FLOWSEEK_ROOT))


def extract_frames(video_path, output_dir, max_frames=None, skip_frames=1):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        max_frames: Maximum number of frames to extract (None = all)
        skip_frames: Extract every Nth frame (1 = all frames, 2 = every other, etc.)
    
    Returns:
        List of extracted frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Skip frames: {skip_frames}")
    
    frame_paths = []
    frame_idx = 0
    extracted = 0
    
    pbar = tqdm(total=min(max_frames or total_frames, total_frames), desc="Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if requested
        if frame_idx % skip_frames == 0:
            frame_path = output_dir / f"frame_{extracted:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            extracted += 1
            pbar.update(1)
            
            if max_frames and extracted >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    print(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths


class LucasKanadeFlow:
    """Lucas-Kanade optical flow"""
    
    @staticmethod
    def compute(img1, img2):
        from scipy.interpolate import griddata
        
        h, w = img1.shape
        y_coords, x_coords = np.mgrid[0:h:10, 0:w:10].astype(np.float32)
        points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
        
        lk_params = {
            'winSize': (15, 15),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, **lk_params)
        
        flow_sparse = next_points - points
        flow_sparse = flow_sparse[status.ravel() == 1]
        points_valid = points[status.ravel() == 1]
        
        flow = np.zeros((h, w, 2), dtype=np.float32)
        
        if len(points_valid) > 0:
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            flow[:, :, 0] = griddata(points_valid, flow_sparse[:, 0], 
                                     (grid_x, grid_y), method='linear', fill_value=0)
            flow[:, :, 1] = griddata(points_valid, flow_sparse[:, 1], 
                                     (grid_x, grid_y), method='linear', fill_value=0)
        
        return flow


class FlowSeekRunner:
    """FlowSeek optical flow"""
    
    def __init__(self, flowseek_root):
        self.flowseek_root = Path(flowseek_root)
        
        import torch
        import json
        from core.flowseek import FlowSeek
        
        # Load config
        config_path = self.flowseek_root / "config/eval/flowseek-T.json"
        with open(config_path, 'r') as f:
            cfg_dict = json.load(f)
        
        # Convert to object
        class ConfigObject:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, ConfigObject(v))
                    else:
                        setattr(self, k, v)
                
                if not hasattr(self, 'use_var'):
                    self.use_var = False
                if not hasattr(self, 'var_max'):
                    self.var_max = 0
                if not hasattr(self, 'var_min'):
                    self.var_min = 0
        
        cfg = ConfigObject(cfg_dict)
        
        # Load model
        print("Loading FlowSeek model...")
        self.model = FlowSeek(cfg)
        
        checkpoint_path = self.flowseek_root / "weights/flowseek_T_TartanCT.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=False)
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()
            print("FlowSeek loaded on GPU")
        else:
            print("FlowSeek loaded on CPU")
        
        self.torch = torch
    
    def compute(self, img1, img2):
        """Compute flow for grayscale or color images"""
        
        # Load as RGB if grayscale
        if len(img1.shape) == 2:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        else:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img1_t = self.torch.from_numpy(img1_rgb).permute(2, 0, 1).float().unsqueeze(0)
        img2_t = self.torch.from_numpy(img2_rgb).permute(2, 0, 1).float().unsqueeze(0)
        
        if self.torch.cuda.is_available():
            img1_t = img1_t.cuda()
            img2_t = img2_t.cuda()
        
        with self.torch.no_grad():
            output = self.model(img1_t, img2_t, iters=32, test_mode=True)
            flow = output['final'][0].cpu().numpy()
        
        return flow.transpose(1, 2, 0)


def flow_to_color(flow):
    """Convert flow to color visualization"""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def create_visualization(frame1, frame2, flow_lk, flow_fs, output_path, frame_idx):
    """Create side-by-side comparison visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Frames and flow magnitude
    axes[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Frame {frame_idx}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Frame {frame_idx + 1}')
    axes[0, 1].axis('off')
    
    # Flow magnitude comparison
    mag_lk = np.sqrt(np.sum(flow_lk**2, axis=2))
    mag_fs = np.sqrt(np.sum(flow_fs**2, axis=2))
    
    axes[0, 2].hist([mag_lk.ravel(), mag_fs.ravel()], bins=50, label=['LK', 'FlowSeek'], alpha=0.7)
    axes[0, 2].set_title('Flow Magnitude Distribution')
    axes[0, 2].set_xlabel('Magnitude (pixels)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].legend()
    
    # Row 2: Flow visualizations
    axes[1, 0].imshow(flow_to_color(flow_lk))
    axes[1, 0].set_title(f'Lucas-Kanade (mean mag: {mag_lk.mean():.2f})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(flow_to_color(flow_fs))
    axes[1, 1].set_title(f'FlowSeek (mean mag: {mag_fs.mean():.2f})')
    axes[1, 1].axis('off')
    
    # Difference
    diff = mag_lk - mag_fs
    im = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 2].set_title('Magnitude Diff (LK - FlowSeek)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_video(args):
    """Main processing function"""
    
    print("=" * 80)
    print("Personal Video Processing: FlowSeek vs Lucas-Kanade")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    
    # Extract frames
    print("\n[1/4] Extracting frames from video...")
    frame_paths = extract_frames(
        args.video,
        frames_dir,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames
    )
    
    if len(frame_paths) < 2:
        print("ERROR: Need at least 2 frames")
        return
    
    # Initialize methods
    print("\n[2/4] Initializing optical flow methods...")
    
    if args.use_flowseek:
        # Change to FlowSeek directory for relative paths
        original_dir = os.getcwd()
        os.chdir(FLOWSEEK_ROOT)
        
        flowseek = FlowSeekRunner(FLOWSEEK_ROOT)
        
        os.chdir(original_dir)
    
    # Process frame pairs
    print(f"\n[3/4] Processing {len(frame_paths)-1} frame pairs...")
    
    num_pairs = min(args.num_pairs, len(frame_paths) - 1) if args.num_pairs else len(frame_paths) - 1
    
    for i in tqdm(range(num_pairs), desc="Computing flow"):
        # Load frames
        frame1 = cv2.imread(str(frame_paths[i]))
        frame2 = cv2.imread(str(frame_paths[i + 1]))
        
        # Convert to grayscale for LK
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute flow
        flow_lk = LucasKanadeFlow.compute(gray1, gray2)
        
        if args.use_flowseek:
            flow_fs = flowseek.compute(frame1, frame2)
        else:
            # Use Farneback as alternative
            flow_fs = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        # Create visualization
        if i < args.visualize_count:
            viz_path = output_dir / f"comparison_{i:03d}.png"
            method_name = "FlowSeek" if args.use_flowseek else "Farneback"
            create_visualization(frame1, frame2, flow_lk, flow_fs, viz_path, i)
    
    print(f"\n[4/4] Results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Process personal videos with optical flow')
    
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output-dir', default='./video_results', help='Output directory')
    parser.add_argument('--use-flowseek', action='store_true', help='Use FlowSeek (otherwise uses Farneback)')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to extract')
    parser.add_argument('--skip-frames', type=int, default=1, help='Extract every Nth frame')
    parser.add_argument('--num-pairs', type=int, default=10, help='Number of frame pairs to process')
    parser.add_argument('--visualize-count', type=int, default=5, help='Number of visualizations to save')
    
    args = parser.parse_args()
    
    process_video(args)


if __name__ == '__main__':
    main()