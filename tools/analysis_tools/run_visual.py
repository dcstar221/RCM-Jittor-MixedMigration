"""
Run visual.py with correct paths for RCM-Fusion test results visualization.
Usage: python tools/analysis_tools/run_visual.py
"""
import os
import sys
import mmcv
from nuscenes.nuscenes import NuScenes

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import visual module - we need to set nusc before calling its functions
import tools.analysis_tools.visual as visual

# ========== Configuration ==========
NUSCENES_VERSION = 'v1.0-mini'
NUSCENES_DATAROOT = os.path.join(project_root, 'data', 'RCM_Data', 'v1.0-mini')
RESULTS_JSON = os.path.join(project_root, 'test', 'rcm-fusion_r50', 'Mon_Feb_23_03_00_21_2026', 'pts_bbox', 'results_nusc.json')
OUTPUT_DIR = os.path.join(project_root, 'test', 'rcm-fusion_r50', 'Mon_Feb_23_03_00_21_2026', 'visualizations')
NUM_SAMPLES = 10  # Number of samples to visualize
# ====================================

def main():
    print(f"[INFO] NuScenes version: {NUSCENES_VERSION}")
    print(f"[INFO] NuScenes dataroot: {NUSCENES_DATAROOT}")
    print(f"[INFO] Results JSON: {RESULTS_JSON}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    # Check paths
    if not os.path.exists(NUSCENES_DATAROOT):
        print(f"[ERROR] NuScenes dataroot not found: {NUSCENES_DATAROOT}")
        return
    if not os.path.exists(RESULTS_JSON):
        print(f"[ERROR] Results JSON not found: {RESULTS_JSON}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize NuScenes
    print("\n[INFO] Loading NuScenes database...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    
    # Set nusc as the global variable in the visual module
    visual.nusc = nusc
    
    # Load results
    print(f"\n[INFO] Loading results from {RESULTS_JSON}...")
    results = mmcv.load(RESULTS_JSON)
    sample_token_list = list(results['results'].keys())
    print(f"[INFO] Total samples in results: {len(sample_token_list)}")
    
    # Filter to only samples that exist in the mini dataset
    valid_sample_tokens = set()
    for sample in nusc.sample:
        valid_sample_tokens.add(sample['token'])
    
    available_tokens = [t for t in sample_token_list if t in valid_sample_tokens]
    print(f"[INFO] Samples available in {NUSCENES_VERSION}: {len(available_tokens)}")
    
    if len(available_tokens) == 0:
        print("[ERROR] No matching sample tokens found between results and NuScenes database!")
        print(f"[DEBUG] First 5 result tokens: {sample_token_list[:5]}")
        print(f"[DEBUG] First 5 nusc sample tokens: {[s['token'] for s in nusc.sample[:5]]}")
        return
    
    # Visualize samples
    num_to_render = min(NUM_SAMPLES, len(available_tokens))
    print(f"\n[INFO] Rendering {num_to_render} samples...")
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving to file
    
    for i in range(num_to_render):
        token = available_tokens[i]
        out_path = os.path.join(OUTPUT_DIR, f'sample_{i:03d}_{token[:8]}')
        print(f"\n[{i+1}/{num_to_render}] Rendering sample: {token}")
        print(f"  -> Output: {out_path}")
        try:
            visual.render_sample_data(
                token, 
                pred_data=results, 
                out_path=out_path,
                verbose=False  # Don't plt.show(), just save
            )
            print(f"  -> Done! Files saved: {out_path}_bev.png, {out_path}_camera.png")
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[INFO] Visualization complete! Output saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
