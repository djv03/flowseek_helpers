#!/usr/bin/env python3
"""
Setup Verification Script
Run this before using the full pipeline to check your environment
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.7+)")
        return False


def check_packages():
    """Check required packages"""
    print("\nChecking required packages...")
    
    required = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas': 'pandas',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_sintel_dataset(sintel_root):
    """Check Sintel dataset structure"""
    print(f"\nChecking Sintel dataset at: {sintel_root}")
    
    sintel_path = Path(sintel_root)
    if not sintel_path.exists():
        print(f"  ✗ Directory not found: {sintel_root}")
        return False
    
    print(f"  ✓ Root directory exists")
    
    # Check training directory
    training_dir = sintel_path / 'training'
    if not training_dir.exists():
        print(f"  ✗ Training directory not found")
        return False
    
    print(f"  ✓ Training directory exists")
    
    # Check subdirectories
    required_dirs = ['clean', 'final', 'flow']
    found_dirs = []
    
    for subdir in required_dirs:
        path = training_dir / subdir
        if path.exists():
            # Count sequences
            sequences = [d for d in path.iterdir() if d.is_dir()]
            print(f"  ✓ {subdir}/ ({len(sequences)} sequences)")
            found_dirs.append(subdir)
        else:
            print(f"  ✗ {subdir}/ not found")
    
    if len(found_dirs) < 3:
        return False
    
    # Check a sample sequence
    clean_dir = training_dir / 'clean'
    flow_dir = training_dir / 'flow'
    
    sequences = sorted([d.name for d in clean_dir.iterdir() if d.is_dir()])
    if sequences:
        sample_seq = sequences[0]
        
        # Count images
        img_count = len(list((clean_dir / sample_seq).glob('*.png')))
        flow_count = len(list((flow_dir / sample_seq).glob('*.flo')))
        
        print(f"\n  Sample sequence: {sample_seq}")
        print(f"    Images: {img_count}")
        print(f"    Flow files: {flow_count}")
        
        if img_count > 0 and flow_count > 0:
            print(f"  ✓ Dataset appears valid")
            return True
    
    print(f"  ✗ Could not verify dataset contents")
    return False


def check_flowseek(flowseek_root):
    """Check FlowSeek installation"""
    print(f"\nChecking FlowSeek at: {flowseek_root}")
    
    if flowseek_root is None:
        print("  ⚠ FlowSeek path not provided (will skip FlowSeek comparison)")
        return None
    
    flowseek_path = Path(flowseek_root)
    if not flowseek_path.exists():
        print(f"  ✗ Directory not found: {flowseek_root}")
        return False
    
    print(f"  ✓ Root directory exists")
    
    # Look for common inference scripts
    script_patterns = ['demo.py', 'inference.py', 'run.py', 'eval.py', 'predict.py']
    found_scripts = []
    
    for pattern in script_patterns:
        matches = list(flowseek_path.glob(f"**/{pattern}"))
        if matches:
            found_scripts.extend(matches)
    
    if found_scripts:
        print(f"  ✓ Found {len(found_scripts)} potential inference script(s)")
        for script in found_scripts[:3]:  # Show first 3
            print(f"    - {script.relative_to(flowseek_path)}")
    else:
        print(f"  ✗ No inference scripts found")
        print(f"    Check FlowSeek README for correct structure")
        return False
    
    # Look for checkpoints
    checkpoint_patterns = ['*.pth', '*.pt', '*.ckpt']
    found_checkpoints = []
    
    for pattern in checkpoint_patterns:
        matches = list(flowseek_path.glob(f"**/{pattern}"))
        if matches:
            found_checkpoints.extend(matches)
    
    if found_checkpoints:
        print(f"  ✓ Found {len(found_checkpoints)} checkpoint file(s)")
    else:
        print(f"  ⚠ No checkpoint files found (may need to download pretrained weights)")
    
    print(f"\n  ⚠ You still need to configure flowseek_helper.py")
    print(f"    Run: python flowseek_helper.py {flowseek_root}")
    
    return True


def check_disk_space(output_dir):
    """Check available disk space"""
    print(f"\nChecking disk space for output: {output_dir}")
    
    try:
        stat = os.statvfs(output_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        if free_gb > 10:
            print(f"  ✓ Available space: {free_gb:.1f} GB")
            return True
        else:
            print(f"  ⚠ Low disk space: {free_gb:.1f} GB")
            print(f"    Results may require several GB")
            return True
    except:
        print(f"  ⚠ Could not check disk space")
        return True


def main():
    print("=" * 80)
    print("Optical Flow Pipeline - Setup Verification")
    print("=" * 80)
    
    # Get paths from command line or defaults
    if len(sys.argv) > 1:
        sintel_root = sys.argv[1]
    else:
        sintel_root = "/proj/ciptmp/we03cyna/my_ml_data/MPI-Sintel-complete"
    
    if len(sys.argv) > 2:
        flowseek_root = sys.argv[2]
    else:
        flowseek_root = "/proj/ciptmp/we03cyna/flowseek_cloned"
    
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    else:
        output_dir = "./results"
    
    # Run checks
    checks = {
        'Python version': check_python_version(),
        'Required packages': check_packages(),
        'Sintel dataset': check_sintel_dataset(sintel_root),
        'FlowSeek': check_flowseek(flowseek_root),
        'Disk space': check_disk_space(output_dir)
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in checks.values() if v is True)
    total = sum(1 for v in checks.values() if v is not None)
    
    for check, result in checks.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        
        print(f"{status:10} {check}")
    
    print("\n" + "=" * 80)
    
    if checks['Python version'] and checks['Required packages'] and checks['Sintel dataset']:
        print("✓ Core requirements met - ready to test Lucas-Kanade")
        print("\nQuick test command:")
        print(f"python optical_flow_comparison.py \\")
        print(f"    --sintel-root {sintel_root} \\")
        print(f"    --sequences alley_1 \\")
        print(f"    --max-pairs 5 \\")
        print(f"    --output-dir {output_dir}")
        
        if checks['FlowSeek']:
            print("\n✓ FlowSeek detected - configure integration next")
            print(f"python flowseek_helper.py {flowseek_root}")
        else:
            print("\n⚠ FlowSeek not ready - will run Lucas-Kanade only")
    else:
        print("✗ Setup incomplete - fix issues above before proceeding")
    
    print("=" * 80)


if __name__ == '__main__':
    main()