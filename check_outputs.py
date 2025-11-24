import os
import numpy as np
import glob

def check_outputs(base_dir="output/embeddings"):
    models = ['openl3', 'crepe', 'madmom', 'mert']
    
    print("=== Checking Output Embeddings ===")
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        files = glob.glob(os.path.join(model_dir, "*.npy"))
        print(f"\nModel: {model} ({len(files)} files found)")
        
        if not files:
            print("  No files found yet.")
            continue
            
        # Check first 3 files
        for f in files[:3]:
            try:
                data = np.load(f)
                print(f"  {os.path.basename(f)}: {data.shape}")
            except Exception as e:
                print(f"  Error loading {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    check_outputs()
