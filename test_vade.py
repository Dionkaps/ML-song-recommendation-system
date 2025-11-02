"""
Quick test script to verify VaDE implementation and integration.
Run this after installing requirements to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  ✗ PyTorch not found - install with: pip install torch>=2.0.0")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError:
        print("  ✗ NumPy not found")
        return False
    
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("  ✗ scikit-learn not found")
        return False
    
    try:
        from src.clustering import vade
        print(f"  ✓ VaDE module loaded")
    except ImportError as e:
        print(f"  ✗ VaDE module import failed: {e}")
        return False
    
    try:
        from src.clustering import kmeans, gmm, hdbscan
        print(f"  ✓ Other clustering modules loaded")
    except ImportError as e:
        print(f"  ✗ Other clustering modules failed: {e}")
        return False
    
    return True


def test_vade_model():
    """Test VaDE model instantiation."""
    print("\nTesting VaDE model...")
    
    try:
        import torch
        from src.clustering.vade import VaDE
        
        # Create a small model
        model = VaDE(
            input_dim=50,
            latent_dim=5,
            enc_hidden=[32, 16],
            dec_hidden=[16, 32],
            n_components=3
        )
        
        print(f"  ✓ Model created successfully")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(10, 50)
        x_recon, mu, logvar, z, gamma = model(x)
        
        assert x_recon.shape == (10, 50), "Reconstruction shape mismatch"
        assert mu.shape == (10, 5), "Latent mean shape mismatch"
        assert gamma.shape == (10, 3), "Gamma shape mismatch"
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input: {x.shape}, Output: {x_recon.shape}")
        print(f"    Latent: {z.shape}, Responsibilities: {gamma.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_loss_computation():
    """Test VaDE loss computation."""
    print("\nTesting loss computation...")
    
    try:
        import torch
        from src.clustering.vade import VaDE, vade_loss
        
        model = VaDE(input_dim=20, latent_dim=3, n_components=2)
        x = torch.randn(5, 20)
        
        x_recon, mu, logvar, z, gamma = model(x)
        loss, stats = vade_loss(
            x, x_recon, mu, logvar, gamma,
            model.pi_logits, model.mu_c, model.logvar_c
        )
        
        print(f"  ✓ Loss computed successfully")
        print(f"    Total: {loss.item():.4f}")
        print(f"    Recon: {stats['recon']:.4f}")
        print(f"    KL_z: {stats['kl_z']:.4f}")
        print(f"    KL_c: {stats['kl_c']:.4f}")
        
        # Check loss is finite
        assert torch.isfinite(loss), "Loss is not finite"
        
        # Test backward pass
        loss.backward()
        print(f"  ✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Loss test failed: {e}")
        return False


def test_integration():
    """Test integration with existing code."""
    print("\nTesting integration...")
    
    try:
        from src.clustering.vade import run_vade_clustering
        from src.clustering.kmeans import build_group_weights, _load_genre_mapping
        
        # Test helper functions work
        weights = build_group_weights(n_mfcc=13, n_mels=128, n_genres=10, include_genre=True)
        print(f"  ✓ build_group_weights works: {len(weights)} dims")
        
        print(f"  ✓ Integration functions accessible")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False


def test_pipeline_integration():
    """Test that VaDE is properly integrated into run_pipeline.py"""
    print("\nTesting pipeline integration...")
    
    try:
        pipeline_path = Path("run_pipeline.py")
        if not pipeline_path.exists():
            print(f"  ✗ run_pipeline.py not found")
            return False
        
        content = pipeline_path.read_text()
        
        # Check for VaDE in choices
        if '"vade"' not in content and "'vade'" not in content:
            print(f"  ✗ VaDE not in clustering method choices")
            return False
        
        print(f"  ✓ VaDE in clustering choices")
        
        # Check for VaDE execution branch
        if "src/clustering/vade.py" not in content:
            print(f"  ✗ VaDE execution branch not found")
            return False
        
        print(f"  ✓ VaDE execution branch present")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Pipeline integration test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("VaDE Implementation Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("VaDE Model", test_vade_model()))
    results.append(("Loss Computation", test_loss_computation()))
    results.append(("Integration", test_integration()))
    results.append(("Pipeline Integration", test_pipeline_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<30} {status}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! VaDE is ready to use.")
        print("\nTo run VaDE clustering:")
        print("  python run_pipeline.py --clustering-method vade")
        print("  or")
        print("  python src/clustering/vade.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
