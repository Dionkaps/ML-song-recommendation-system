"""
Test script to verify all clustering algorithms work properly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import traceback


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    failures = []
    
    # Core dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        failures.append("PyTorch")
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        failures.append("scikit-learn")
    
    try:
        import hdbscan
        try:
            print(f"✓ HDBSCAN {hdbscan.__version__}")
        except AttributeError:
            print(f"✓ HDBSCAN (version unavailable)")
    except ImportError as e:
        print(f"✗ HDBSCAN import failed: {e}")
        failures.append("HDBSCAN")
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        failures.append("pandas")
    
    # Clustering modules
    try:
        from src.clustering import kmeans
        print(f"✓ KMeans module loaded")
    except ImportError as e:
        print(f"✗ KMeans module import failed: {e}")
        failures.append("KMeans module")
    
    try:
        from src.clustering import gmm
        print(f"✓ GMM module loaded")
    except ImportError as e:
        print(f"✗ GMM module import failed: {e}")
        failures.append("GMM module")
    
    try:
        from src.clustering import hdbscan as hdbscan_module
        print(f"✓ HDBSCAN module loaded")
    except ImportError as e:
        print(f"✗ HDBSCAN module import failed: {e}")
        failures.append("HDBSCAN module")
    
    try:
        from src.clustering import vade
        print(f"✓ VaDE module loaded")
    except ImportError as e:
        print(f"✗ VaDE module import failed: {e}")
        failures.append("VaDE module")
    
    try:
        from config import feature_vars
        print(f"✓ Config module loaded")
    except ImportError as e:
        print(f"✗ Config module import failed: {e}")
        failures.append("Config")
    
    print()
    if failures:
        print(f"⚠ Import failures: {', '.join(failures)}")
        return False
    else:
        print("✓ All imports successful!")
        return True


def check_data_availability():
    """Check if required data files exist."""
    print("\n" + "=" * 70)
    print("CHECKING DATA AVAILABILITY")
    print("=" * 70)
    
    # Check for genre data
    genres_dir = PROJECT_ROOT / "genres_original"
    if not genres_dir.exists():
        print(f"✗ Genre data directory not found: {genres_dir}")
        return False
    
    genre_folders = list(genres_dir.iterdir())
    print(f"✓ Genre data directory exists: {genres_dir}")
    print(f"  Found {len([f for f in genre_folders if f.is_dir()])} genre folders")
    
    # Check for extracted features
    results_dir = PROJECT_ROOT / "output" / "results"
    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        print(f"  Run feature extraction first: python src/features/extract_features.py")
        return False
    
    # Count feature files
    feature_files = list(results_dir.glob("*.npy"))
    print(f"✓ Results directory exists: {results_dir}")
    print(f"  Found {len(feature_files)} feature files")
    
    if len(feature_files) == 0:
        print(f"✗ No feature files found - run feature extraction first")
        return False
    
    return True


def test_kmeans():
    """Test KMeans clustering."""
    print("\n" + "=" * 70)
    print("TESTING KMEANS CLUSTERING")
    print("=" * 70)
    
    try:
        from src.clustering import kmeans
        
        print("Running KMeans clustering...")
        kmeans.run_kmeans_clustering(
            audio_dir="genres_original",
            results_dir="output/results",
            n_clusters=10,
            dynamic_cluster_selection=False,
        )
        print("✓ KMeans clustering completed successfully!")
        
        # Check if output file was created
        output_file = PROJECT_ROOT / "output" / "audio_clustering_results_kmeans.csv"
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"  Clustered {len(df)} files into {df['Cluster'].nunique()} clusters")
            return True
        else:
            print(f"✗ Output file not found: {output_file}")
            return False
            
    except Exception as e:
        print(f"✗ KMeans clustering failed: {e}")
        traceback.print_exc()
        return False


def test_gmm():
    """Test GMM clustering."""
    print("\n" + "=" * 70)
    print("TESTING GMM CLUSTERING")
    print("=" * 70)
    
    try:
        from src.clustering import gmm
        
        print("Running GMM clustering...")
        gmm.run_gmm_clustering(
            audio_dir="genres_original",
            results_dir="output/results",
            n_components=10,
            covariance_type="full",
            dynamic_component_selection=False,
        )
        print("✓ GMM clustering completed successfully!")
        
        # Check if output file was created
        output_file = PROJECT_ROOT / "output" / "audio_clustering_results_gmm.csv"
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"  Clustered {len(df)} files into {df['Cluster'].nunique()} clusters")
            return True
        else:
            print(f"✗ Output file not found: {output_file}")
            return False
            
    except Exception as e:
        print(f"✗ GMM clustering failed: {e}")
        traceback.print_exc()
        return False


def test_hdbscan():
    """Test HDBSCAN clustering."""
    print("\n" + "=" * 70)
    print("TESTING HDBSCAN CLUSTERING")
    print("=" * 70)
    
    try:
        from src.clustering import hdbscan as hdbscan_module
        
        print("Running HDBSCAN clustering...")
        hdbscan_module.run_hdbscan_clustering(
            audio_dir="genres_original",
            results_dir="output/results",
            min_cluster_size=10,
        )
        print("✓ HDBSCAN clustering completed successfully!")
        
        # Check if output file was created
        output_file = PROJECT_ROOT / "output" / "audio_clustering_results_hdbscan.csv"
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            import pandas as pd
            df = pd.read_csv(output_file)
            clusters = df['Cluster'].unique()
            noise_points = (df['Cluster'] == -1).sum() if -1 in clusters else 0
            valid_clusters = len([c for c in clusters if c != -1])
            print(f"  Clustered {len(df)} files into {valid_clusters} clusters")
            if noise_points > 0:
                print(f"  Noise points: {noise_points}")
            return True
        else:
            print(f"✗ Output file not found: {output_file}")
            return False
            
    except Exception as e:
        print(f"✗ HDBSCAN clustering failed: {e}")
        traceback.print_exc()
        return False


def test_vade():
    """Test VaDE clustering."""
    print("\n" + "=" * 70)
    print("TESTING VADE CLUSTERING")
    print("=" * 70)
    
    try:
        from src.clustering import vade
        
        print("Running VaDE clustering (this may take a few minutes)...")
        vade.run_vade_clustering(
            audio_dir="genres_original",
            results_dir="output/results",
            n_components=10,
            latent_dim=10,
            pretrain_epochs=20,
            train_epochs=50,  # Reduced for quick testing
        )
        print("✓ VaDE clustering completed successfully!")
        
        # Check if output file was created
        output_file = PROJECT_ROOT / "output" / "audio_clustering_results_vade.csv"
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"  Clustered {len(df)} files into {df['Cluster'].nunique()} clusters")
            return True
        else:
            print(f"✗ Output file not found: {output_file}")
            return False
            
    except Exception as e:
        print(f"✗ VaDE clustering failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CLUSTERING ALGORITHMS TEST SUITE")
    print("=" * 70)
    
    results = {
        "Imports": False,
        "Data": False,
        "KMeans": False,
        "GMM": False,
        "HDBSCAN": False,
        "VaDE": False,
    }
    
    # Test imports
    results["Imports"] = test_imports()
    if not results["Imports"]:
        print("\n⚠ Cannot proceed - import failures detected")
        print("Please install missing dependencies from requirements.txt")
        return
    
    # Check data
    results["Data"] = check_data_availability()
    if not results["Data"]:
        print("\n⚠ Cannot proceed - required data not found")
        print("Please run feature extraction first:")
        print("  python src/features/extract_features.py")
        return
    
    # Test each clustering algorithm
    results["KMeans"] = test_kmeans()
    results["GMM"] = test_gmm()
    results["HDBSCAN"] = test_hdbscan()
    results["VaDE"] = test_vade()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len([v for v in results.values() if v])
    print(f"\nTotal: {total}/{len(results)} tests passed")
    
    if all(results.values()):
        print("\n🎉 All tests passed! All clustering algorithms are working properly.")
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    main()
