"""Compare BIC and AIC scores for GMM component selection."""
import pandas as pd
from pathlib import Path
import os
import sys

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

def compare_bic_aic():
    """Display and compare BIC vs AIC model selection criteria."""
    csv_path = Path("output/metrics/gmm_selection_criteria.csv")
    
    if not csv_path.exists():
        print("❌ File not found. Run GMM clustering first:")
        print("   python src/clustering/gmm.py")
        return
    
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("GMM Model Selection: BIC vs AIC Comparison")
    print("=" * 70)
    print("\nNote: Lower (more negative) values are better for both BIC and AIC\n")
    
    # Display table
    print(df.to_string(index=False))
    
    # Find best models
    bic_best_idx = df['BIC'].idxmin()
    aic_best_idx = df['AIC'].idxmin()
    
    bic_best = df.loc[bic_best_idx]
    aic_best = df.loc[aic_best_idx]
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"BIC recommends: {int(bic_best['Components'])} components")
    print(f"  → BIC = {bic_best['BIC']:.2f}")
    print(f"\nAIC recommends: {int(aic_best['Components'])} components")
    print(f"  → AIC = {aic_best['AIC']:.2f}")
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("BIC (Bayesian Information Criterion):")
    print("  • Penalizes model complexity more heavily")
    print("  • Tends to select simpler models (fewer components)")
    print("  • Better for avoiding overfitting")
    print("  • Preferred when interpretability matters")
    
    print("\nAIC (Akaike Information Criterion):")
    print("  • Less penalty for model complexity")
    print("  • May select more complex models (more components)")
    print("  • Better for predictive accuracy")
    print("  • Preferred when prediction is the main goal")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if bic_best['Components'] == aic_best['Components']:
        print(f"✓ Both metrics AGREE: Use {int(bic_best['Components'])} components")
        print("  This is a strong signal for model selection!")
    else:
        print(f"⚠ Metrics DISAGREE:")
        print(f"  • BIC suggests {int(bic_best['Components'])} components (simpler)")
        print(f"  • AIC suggests {int(aic_best['Components'])} components (more complex)")
        print("\n  Consider your goal:")
        print("    - Interpretability/simplicity → Use BIC recommendation")
        print("    - Prediction/detailed clustering → Use AIC recommendation")
        print("    - Middle ground → Average of both or run evaluation metrics")
    
    print("=" * 70)

if __name__ == "__main__":
    compare_bic_aic()
