#!/usr/bin/env python3
"""
Optical Flow Comparison Pipeline: FlowSeek vs Lucas-Kanade
Focus: Low-texture region performance analysis on MPI-Sintel dataset
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class SintelDataLoader:
    """Load MPI-Sintel dataset with ground truth flow"""
    
    def __init__(self, sintel_root: str, pass_type: str = 'clean'):
        """
        Args:
            sintel_root: Path to MPI-Sintel-complete directory
            pass_type: 'clean' or 'final'
        """
        self.sintel_root = Path(sintel_root)
        self.pass_type = pass_type
        
        self.img_dir = self.sintel_root / 'training' / pass_type
        self.flow_dir = self.sintel_root / 'training' / 'flow'
        
        # Verify directories exist
        if not self.img_dir.exists():
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not self.flow_dir.exists():
            raise ValueError(f"Flow directory not found: {self.flow_dir}")
    
    def get_sequences(self) -> List[str]:
        """Get list of available sequences"""
        return sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
    
    def load_flow(self, flow_path: Path) -> np.ndarray:
        """Load .flo file (Sintel ground truth format)"""
        with open(flow_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if magic != 202021.25:
                raise ValueError(f'Invalid .flo file: {flow_path}')
            w, h = np.fromfile(f, np.int32, count=2)
            flow = np.fromfile(f, np.float32, count=2*w*h)
            flow = flow.reshape((h, w, 2))
        return flow
    
    def get_frame_pairs(self, sequence: str) -> List[Dict]:
        """Get all consecutive frame pairs for a sequence with ground truth"""
        seq_img_dir = self.img_dir / sequence
        seq_flow_dir = self.flow_dir / sequence
        
        img_files = sorted(seq_img_dir.glob('*.png'))
        pairs = []
        
        for i in range(len(img_files) - 1):
            frame_num = int(img_files[i].stem.split('_')[-1])
            flow_file = seq_flow_dir / f'frame_{frame_num:04d}.flo'
            
            if flow_file.exists():
                pairs.append({
                    'img1': img_files[i],
                    'img2': img_files[i + 1],
                    'flow_gt': flow_file,
                    'sequence': sequence,
                    'frame': frame_num
                })
        
        return pairs


class LowTextureDetector:
    """Detect low-texture regions in images"""
    
    @staticmethod
    def compute_texture_map(image: np.ndarray, window_size: int = 15) -> np.ndarray:
        """
        Compute texture strength map using local variance
        
        Args:
            image: Grayscale image
            window_size: Size of local window for variance computation
        
        Returns:
            Texture strength map (higher = more texture)
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Compute local mean
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        local_mean = cv2.filter2D(img_float, -1, kernel)
        
        # Compute local variance
        local_sq_mean = cv2.filter2D(img_float ** 2, -1, kernel)
        local_var = local_sq_mean - local_mean ** 2
        
        return np.sqrt(np.maximum(local_var, 0))
    
    @staticmethod
    def get_low_texture_mask(image: np.ndarray, 
                            percentile: float = 25,
                            window_size: int = 15,
                            min_region_size: int = 100) -> np.ndarray:
        """
        Create binary mask for low-texture regions (adaptive thresholding)
        
        Args:
            image: Grayscale image
            percentile: Percentile threshold for "low texture" (lower = stricter)
            window_size: Size of local window for variance computation
            min_region_size: Minimum size of low-texture regions to keep
        
        Returns:
            Binary mask (1 = low texture, 0 = high texture)
        """
        texture_map = LowTextureDetector.compute_texture_map(image, window_size)
        
        # Adaptive threshold based on percentile
        threshold = np.percentile(texture_map, percentile)
        
        # Create binary mask
        mask = (texture_map < threshold).astype(np.uint8)
        
        # Remove small isolated regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        # Filter by size
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_region_size:
                mask[labels == label] = 0
        
        return mask


class LucasKanadeFlow:
    """Lucas-Kanade optical flow implementation"""
    
    @staticmethod
    def compute(img1: np.ndarray, img2: np.ndarray, **params) -> np.ndarray:
        """
        Compute Lucas-Kanade optical flow
        
        Args:
            img1, img2: Consecutive frames (grayscale)
            **params: Parameters for cv2.calcOpticalFlowPyrLK
        
        Returns:
            Dense flow field (H x W x 2)
        """
        # Default LK parameters
        lk_params = {
            'winSize': (15, 15),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        lk_params.update(params)
        
        # Generate dense grid of points
        h, w = img1.shape
        y_coords, x_coords = np.mgrid[0:h:10, 0:w:10].astype(np.float32)
        points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
        
        # Compute sparse flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            img1, img2, points, None, **lk_params
        )
        
        # Convert sparse to dense using interpolation
        flow_sparse = next_points - points
        flow_sparse = flow_sparse[status.ravel() == 1]
        points_valid = points[status.ravel() == 1]
        
        # Dense flow field via interpolation
        flow = np.zeros((h, w, 2), dtype=np.float32)
        
        if len(points_valid) > 0:
            from scipy.interpolate import griddata
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            
            flow[:, :, 0] = griddata(points_valid, flow_sparse[:, 0], 
                                     (grid_x, grid_y), method='linear', fill_value=0)
            flow[:, :, 1] = griddata(points_valid, flow_sparse[:, 1], 
                                     (grid_x, grid_y), method='linear', fill_value=0)
        
        return flow


class FlowSeekRunner:
    """Interface to run FlowSeek model"""
    
    def __init__(self, flowseek_root: str, config_path: str = None, checkpoint_path: str = None):
        """
        Args:
            flowseek_root: Path to FlowSeek repository
            config_path: Path to FlowSeek config (optional)
            checkpoint_path: Path to FlowSeek checkpoint (optional)
        """
        self.flowseek_root = Path(flowseek_root)
        
        if not self.flowseek_root.exists():
            raise ValueError(f"FlowSeek directory not found: {flowseek_root}")
        
        # Set default paths
        self.config_path = config_path or str(self.flowseek_root / "config/eval/flowseek-T.json")
        self.checkpoint_path = checkpoint_path or str(self.flowseek_root / "weights/flowseek_T_TartanCT.pth")
        
        # Check if files exist
        if not Path(self.checkpoint_path).exists():
            # Try alternative weight names
            alt_weights = list((self.flowseek_root / "weights").glob("flowseek*.pth"))
            if alt_weights:
                self.checkpoint_path = str(alt_weights[0])
                print(f"[FlowSeek] Using checkpoint: {self.checkpoint_path}")
            else:
                raise FileNotFoundError(f"No FlowSeek weights found in {self.flowseek_root / 'weights'}")
        
        print(f"[FlowSeek] Config: {self.config_path}")
        print(f"[FlowSeek] Checkpoint: {self.checkpoint_path}")
        
        # Add FlowSeek to path
        sys.path.insert(0, str(self.flowseek_root))
        
        # Initialize model (load once, reuse for all pairs)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FlowSeek model once"""
        try:
            # Import FlowSeek
            try:
                from core.flowseek import FlowSeek
            except:
                from flowseek import FlowSeek
            
            import json
            import torch
            
            # Load config
            with open(self.config_path, 'r') as f:
                cfg_dict = json.load(f)
            
            # Convert dict to object with attributes (FlowSeek expects this)
            class ConfigObject:
                def __init__(self, d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            setattr(self, k, ConfigObject(v))
                        else:
                            setattr(self, k, v)
                    
                    # Add missing attributes with defaults (from flowseek.py)
                    if not hasattr(self, 'use_var'):
                        self.use_var = False
                    if not hasattr(self, 'var_max'):
                        self.var_max = 0
                    if not hasattr(self, 'var_min'):
                        self.var_min = 0
            
            cfg = ConfigObject(cfg_dict)
            
            # Initialize model
            print("[FlowSeek] Initializing model (this may take a moment)...")
            self.model = FlowSeek(cfg)
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.cuda()
                print("[FlowSeek] Model loaded on GPU")
            else:
                print("[FlowSeek] Model loaded on CPU (will be slow)")
            
            # Store torch for later use
            self.torch = torch
            
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize FlowSeek: {e}")
            print(f"[ERROR] Make sure FlowSeek dependencies are installed")
            print(f"[ERROR] Try: cd {self.flowseek_root} && pip install -r requirements.txt")
            raise
    
    def compute(self, img1_path: Path, img2_path: Path, 
                output_path: Path) -> np.ndarray:
        """
        Run FlowSeek on image pair
        
        Args:
            img1_path, img2_path: Paths to consecutive frames
            output_path: Where to save output flow
        
        Returns:
            Flow field (H x W x 2)
        """
        try:
            from PIL import Image
            
            # Load images
            def load_image(imfile):
                img = Image.open(imfile).convert('RGB')
                img = np.array(img).astype(np.uint8)
                img = self.torch.from_numpy(img).permute(2, 0, 1).float()
                return img
            
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)
            
            # Add batch dimension
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            
            if self.torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()
            
            # Run inference
            with self.torch.no_grad():
                # FlowSeek returns a dict: {'final': ..., 'flow': [...], 'info': [...], 'nf': ...}
                output = self.model(img1, img2, iters=32, test_mode=True)
                
                # Get the final flow prediction
                flow = output['final'][0].cpu().numpy()  # [2, H, W]
            
            # Convert from (2, H, W) to (H, W, 2)
            flow = flow.transpose(1, 2, 0)
            
            # Save as .flo
            self._write_flow(output_path, flow)
            
            return flow
            
        except Exception as e:
            print(f"\n[ERROR] FlowSeek inference failed: {e}")
            raise RuntimeError(f"FlowSeek failed: {e}")
    
    def _write_flow(self, filename, uv):
        """Write optical flow in .flo format"""
        with open(filename, 'wb') as f:
            np.array([202021.25], dtype=np.float32).tofile(f)
            np.array([uv.shape[1], uv.shape[0]], dtype=np.int32).tofile(f)
            uv.astype(np.float32).tofile(f)
    
    def load_flow_output(self, path: Path) -> np.ndarray:
        """Load FlowSeek output - update based on actual format"""
        # Placeholder - update based on FlowSeek's output format
        # Could be .flo, .npy, .png, etc.
        if path.suffix == '.flo':
            return SintelDataLoader(None).load_flow(path)
        elif path.suffix == '.npy':
            return np.load(path)
        else:
            raise NotImplementedError(f"Unknown flow format: {path.suffix}")


class FlowEvaluator:
    """Evaluate optical flow predictions"""
    
    @staticmethod
    def compute_epe(flow_pred: np.ndarray, flow_gt: np.ndarray, 
                    mask: Optional[np.ndarray] = None) -> float:
        """
        Compute End-Point Error
        
        Args:
            flow_pred: Predicted flow (H x W x 2)
            flow_gt: Ground truth flow (H x W x 2)
            mask: Optional binary mask to compute EPE only in specific regions
        
        Returns:
            Mean EPE
        """
        # Compute L2 distance
        epe = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))
        
        if mask is not None:
            epe = epe[mask > 0]
        
        return np.mean(epe)
    
    @staticmethod
    def compute_metrics(flow_pred: np.ndarray, flow_gt: np.ndarray,
                       low_texture_mask: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive metrics
        
        Returns:
            Dictionary with overall and low-texture region metrics
        """
        # Overall EPE
        epe_overall = FlowEvaluator.compute_epe(flow_pred, flow_gt)
        
        # Low-texture region EPE
        epe_low_texture = FlowEvaluator.compute_epe(flow_pred, flow_gt, low_texture_mask)
        
        # High-texture region EPE (for comparison)
        high_texture_mask = 1 - low_texture_mask
        epe_high_texture = FlowEvaluator.compute_epe(flow_pred, flow_gt, high_texture_mask)
        
        # Outlier percentage (EPE > 3 pixels and > 5% of magnitude)
        epe_map = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))
        flow_mag = np.sqrt(np.sum(flow_gt ** 2, axis=2))
        outliers = (epe_map > 3) & (epe_map > 0.05 * flow_mag)
        outlier_pct = 100 * np.mean(outliers)
        
        # Low-texture outliers
        outliers_low_texture = outliers[low_texture_mask > 0]
        outlier_pct_low_texture = 100 * np.mean(outliers_low_texture) if len(outliers_low_texture) > 0 else 0
        
        return {
            'epe_overall': float(epe_overall),
            'epe_low_texture': float(epe_low_texture),
            'epe_high_texture': float(epe_high_texture),
            'outlier_pct_overall': float(outlier_pct),
            'outlier_pct_low_texture': float(outlier_pct_low_texture),
            'low_texture_pixel_ratio': float(np.mean(low_texture_mask))
        }


def run_comparison_pipeline(args):
    """Main pipeline execution"""
    
    print("=" * 80)
    print("Optical Flow Comparison: FlowSeek vs Lucas-Kanade")
    print("Hypothesis: FlowSeek > LK in low-texture regions")
    print("=" * 80)
    
    # Initialize components
    print("\n[1/6] Initializing dataset loader...")
    data_loader = SintelDataLoader(args.sintel_root, args.pass_type)
    sequences = data_loader.get_sequences()
    print(f"Found {len(sequences)} sequences: {sequences}")
    
    # Filter sequences if specified
    if args.sequences:
        sequences = [s for s in sequences if s in args.sequences]
        print(f"Testing on {len(sequences)} sequences: {sequences}")
    
    # Initialize methods
    print("\n[2/6] Initializing optical flow methods...")
    if args.use_flowseek:
        flowseek = FlowSeekRunner(args.flowseek_root)
        print(f"FlowSeek loaded from: {args.flowseek_root}")
    
    # Collect all frame pairs
    print("\n[3/6] Collecting frame pairs...")
    all_pairs = []
    for seq in sequences:
        pairs = data_loader.get_frame_pairs(seq)
        # Take only the FIRST pair from each sequence for diversity
        if pairs:
            all_pairs.append(pairs[0])
            print(f"  {seq}: selected 1 pair (frame {pairs[0]['frame']})")
        else:
            print(f"  {seq}: no pairs found")
    
    print(f"Total: {len(all_pairs)} frame pairs (1 from each sequence)")
    print(f"Note: Taking 1 frame pair from each sequence for diverse test cases\n")
    
    # Process each pair
    print("\n[4/6] Processing frame pairs...")
    results = []
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair_idx, pair in enumerate(tqdm(all_pairs, desc="Processing pairs")):
        # Load images
        img1 = cv2.imread(str(pair['img1']), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(pair['img2']), cv2.IMREAD_GRAYSCALE)
        flow_gt = data_loader.load_flow(pair['flow_gt'])
        
        # Detect low-texture regions
        low_texture_mask = LowTextureDetector.get_low_texture_mask(
            img1, 
            percentile=args.texture_percentile,
            window_size=args.texture_window_size
        )
        
        # Compute Lucas-Kanade flow
        flow_lk = LucasKanadeFlow.compute(img1, img2)
        
        # Compute FlowSeek flow (if enabled)
        if args.use_flowseek:
            temp_output = output_dir / f"flowseek_temp_{pair_idx}.flo"
            flow_flowseek = flowseek.compute(pair['img1'], pair['img2'], temp_output)
        else:
            flow_flowseek = None
        
        # Evaluate
        metrics_lk = FlowEvaluator.compute_metrics(flow_lk, flow_gt, low_texture_mask)
        metrics_lk['method'] = 'Lucas-Kanade'
        metrics_lk['sequence'] = pair['sequence']
        metrics_lk['frame'] = pair['frame']
        results.append(metrics_lk)
        
        if flow_flowseek is not None:
            metrics_flowseek = FlowEvaluator.compute_metrics(flow_flowseek, flow_gt, low_texture_mask)
            metrics_flowseek['method'] = 'FlowSeek'
            metrics_flowseek['sequence'] = pair['sequence']
            metrics_flowseek['frame'] = pair['frame']
            results.append(metrics_flowseek)
        
        # Save visualization for first few pairs
        if pair_idx < args.visualize_count:
            save_visualization(pair, img1, img2, flow_gt, flow_lk, flow_flowseek,
                             low_texture_mask, output_dir, pair_idx)
    
    # Save results
    print("\n[5/6] Saving results...")
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Analyze and visualize
    print("\n[6/6] Analyzing results...")
    analyze_results(results, output_dir)
    
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


def save_visualization(pair, img1, img2, flow_gt, flow_lk, flow_flowseek,
                      low_texture_mask, output_dir, idx):
    """Save visualization of results"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Images and low-texture mask
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('Frame 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('Frame 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(low_texture_mask, cmap='RdYlGn_r')
    axes[0, 2].set_title(f'Low-Texture Regions ({100*np.mean(low_texture_mask):.1f}%)')
    axes[0, 2].axis('off')
    
    # Row 2: Ground truth and predictions
    flow_vis_gt = flow_to_color(flow_gt)
    axes[1, 0].imshow(flow_vis_gt)
    axes[1, 0].set_title('Ground Truth Flow')
    axes[1, 0].axis('off')
    
    flow_vis_lk = flow_to_color(flow_lk)
    axes[1, 1].imshow(flow_vis_lk)
    axes[1, 1].set_title('Lucas-Kanade')
    axes[1, 1].axis('off')
    
    if flow_flowseek is not None:
        flow_vis_flowseek = flow_to_color(flow_flowseek)
        axes[1, 2].imshow(flow_vis_flowseek)
        axes[1, 2].set_title('FlowSeek')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
    
    # Row 3: Error maps
    epe_lk = np.sqrt(np.sum((flow_lk - flow_gt) ** 2, axis=2))
    im1 = axes[2, 0].imshow(epe_lk, cmap='hot', vmin=0, vmax=10)
    axes[2, 0].set_title(f'LK Error (mean={np.mean(epe_lk):.2f})')
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0])
    
    if flow_flowseek is not None:
        epe_flowseek = np.sqrt(np.sum((flow_flowseek - flow_gt) ** 2, axis=2))
        im2 = axes[2, 1].imshow(epe_flowseek, cmap='hot', vmin=0, vmax=10)
        axes[2, 1].set_title(f'FlowSeek Error (mean={np.mean(epe_flowseek):.2f})')
        axes[2, 1].axis('off')
        plt.colorbar(im2, ax=axes[2, 1])
        
        # Difference map
        diff = epe_lk - epe_flowseek
        im3 = axes[2, 2].imshow(diff, cmap='RdBu_r', vmin=-5, vmax=5)
        axes[2, 2].set_title('Error Diff (LK - FlowSeek)')
        axes[2, 2].axis('off')
        plt.colorbar(im3, ax=axes[2, 2])
    else:
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
    
    plt.suptitle(f"{pair['sequence']} - Frame {pair['frame']}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'visualization_{idx:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def flow_to_color(flow):
    """Convert flow to color visualization (HSV encoding)"""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def analyze_results(results, output_dir):
    """Analyze and visualize comparison results"""
    
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = df.groupby('method').agg({
        'epe_overall': ['mean', 'std', 'median'],
        'epe_low_texture': ['mean', 'std', 'median'],
        'epe_high_texture': ['mean', 'std', 'median'],
        'outlier_pct_overall': ['mean', 'std'],
        'outlier_pct_low_texture': ['mean', 'std']
    })
    
    print("\n" + str(summary))
    
    # Hypothesis test
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 80)
    
    if 'FlowSeek' in df['method'].values and 'Lucas-Kanade' in df['method'].values:
        flowseek_epe = df[df['method'] == 'FlowSeek']['epe_low_texture'].values
        lk_epe = df[df['method'] == 'Lucas-Kanade']['epe_low_texture'].values
        
        # Paired t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(lk_epe, flowseek_epe)
        
        improvement = 100 * (1 - flowseek_epe.mean() / lk_epe.mean())
        
        print(f"\nLow-Texture Region EPE:")
        print(f"  Lucas-Kanade: {lk_epe.mean():.4f} ± {lk_epe.std():.4f}")
        print(f"  FlowSeek:     {flowseek_epe.mean():.4f} ± {flowseek_epe.std():.4f}")
        print(f"  Improvement:  {improvement:.2f}%")
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")
        
        if improvement > 0 and p_value < 0.05:
            print(f"\n✓ HYPOTHESIS CONFIRMED: FlowSeek significantly outperforms")
            print(f"  Lucas-Kanade in low-texture regions (p<0.05)")
        else:
            print(f"\n✗ HYPOTHESIS NOT CONFIRMED")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # EPE comparison
    df_plot = df.melt(id_vars=['method'], 
                     value_vars=['epe_overall', 'epe_low_texture', 'epe_high_texture'],
                     var_name='region', value_name='EPE')
    
    sns.boxplot(data=df_plot, x='region', y='EPE', hue='method', ax=axes[0, 0])
    axes[0, 0].set_title('EPE Comparison by Region')
    axes[0, 0].set_ylabel('End-Point Error (pixels)')
    axes[0, 0].legend(title='Method')
    
    # Low-texture focus
    sns.violinplot(data=df, x='method', y='epe_low_texture', ax=axes[0, 1])
    axes[0, 1].set_title('Low-Texture Region EPE Distribution')
    axes[0, 1].set_ylabel('EPE (pixels)')
    
    # Per-sequence breakdown
    if 'sequence' in df.columns:
        seq_summary = df.groupby(['sequence', 'method'])['epe_low_texture'].mean().reset_index()
        seq_pivot = seq_summary.pivot(index='sequence', columns='method', values='epe_low_texture')
        seq_pivot.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Low-Texture EPE by Sequence')
        axes[1, 0].set_ylabel('Mean EPE (pixels)')
        axes[1, 0].legend(title='Method')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Outlier comparison
    df_outliers = df.melt(id_vars=['method'],
                         value_vars=['outlier_pct_overall', 'outlier_pct_low_texture'],
                         var_name='region', value_name='outlier_pct')
    
    sns.barplot(data=df_outliers, x='region', y='outlier_pct', hue='method', ax=axes[1, 1])
    axes[1, 1].set_title('Outlier Percentage')
    axes[1, 1].set_ylabel('Outliers (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis plots saved to: {output_dir / 'analysis_plots.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare FlowSeek vs Lucas-Kanade on MPI-Sintel dataset'
    )
    
    # Dataset arguments
    parser.add_argument('--sintel-root', type=str, required=True,
                       help='Path to MPI-Sintel-complete directory')
    parser.add_argument('--pass-type', type=str, default='clean',
                       choices=['clean', 'final'],
                       help='Sintel pass type')
    parser.add_argument('--sequences', nargs='+', default=None,
                       help='Specific sequences to test (default: all)')
    
    # FlowSeek arguments
    parser.add_argument('--flowseek-root', type=str, default=None,
                       help='Path to FlowSeek repository')
    parser.add_argument('--use-flowseek', action='store_true',
                       help='Include FlowSeek in comparison')
    
    # Low-texture detection arguments
    parser.add_argument('--texture-percentile', type=float, default=25,
                       help='Percentile threshold for low-texture (default: 25)')
    parser.add_argument('--texture-window-size', type=int, default=15,
                       help='Window size for texture computation (default: 15)')
    
    # Processing arguments
    parser.add_argument('--max-pairs', type=int, default=None,
                       help='Maximum number of frame pairs to process')
    parser.add_argument('--visualize-count', type=int, default=5,
                       help='Number of pairs to visualize (default: 5)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./flow_comparison_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_flowseek and not args.flowseek_root:
        parser.error('--flowseek-root required when --use-flowseek is set')
    
    # Run pipeline
    run_comparison_pipeline(args)


if __name__ == '__main__':
    main()