# Feature Weight Configuration Guide

## Overview
You can now control how much each feature group influences clustering by editing the global configuration in `config/feature_vars.py`.

## Location
All feature weight settings are in:
```
config/feature_vars.py
```

## Configuration Variable
```python
feature_group_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
```

## Feature Groups (in order)
| Index | Feature Group | Description |
|-------|---------------|-------------|
| 0 | **MFCC** | Mel-frequency cepstral coefficients - captures timbre and texture |
| 1 | **Mel Spectrogram** | Frequency content over time - captures pitch and harmony |
| 2 | **Spectral Centroid** | Brightness of sound - distinguishes bright vs dark sounds |
| 3 | **Zero Crossing Rate** | Noisiness/percussiveness - captures rhythm and percussion |
| 4 | **Genre** | Actual genre labels (only used if `include_genre = True`) |

## How Weights Work
- **Higher value** = More influence on clustering
- **Lower value** = Less influence on clustering
- **1.0** = Default/equal contribution
- **0.0** = Feature is effectively ignored

## Examples

### Example 1: Equal Contribution (Default)
```python
feature_group_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
```
All features contribute equally to clustering.

### Example 2: Prioritize Timbre (MFCC)
```python
feature_group_weights = [3.0, 1.0, 1.0, 1.0, 1.0]
```
MFCC contributes 3x more than other features. Good for grouping by vocal/instrument texture.

### Example 3: Focus on Melody and Brightness
```python
feature_group_weights = [0.5, 2.0, 2.0, 0.5, 1.0]
```
Emphasizes frequency content and brightness, reduces timbre and rhythm.

### Example 4: Rhythm-Focused Clustering
```python
feature_group_weights = [0.5, 0.5, 0.5, 3.0, 1.0]
```
Zero crossing rate dominates - groups songs by percussive/rhythmic patterns.

### Example 5: Ignore Specific Features
```python
feature_group_weights = [1.0, 1.0, 0.0, 0.0, 1.0]
```
Completely ignores spectral centroid and zero crossing rate.

## How to Apply

### Step 1: Edit config/feature_vars.py
```python
# Open the file and modify:
feature_group_weights = [2.0, 1.0, 0.5, 0.5, 1.0]  # Your custom weights
```

### Step 2: Run Any Clustering Algorithm
The weights are **automatically applied** to ALL clustering methods:

```bash
# K-Means
python src/clustering/kmeans.py

# HDBSCAN
python src/clustering/hdbscan.py

# GMM (Gaussian Mixture Model)
python src/clustering/gmm.py

# Or use the pipeline
python run_pipeline.py --clustering-method kmeans
```

**No code changes needed** - just edit the config file once!

## Tips for Tuning

1. **Start with default** `[1.0, 1.0, 1.0, 1.0, 1.0]` and adjust incrementally
2. **Double/halve values** rather than making extreme changes (e.g., try 2.0 or 0.5 first)
3. **Run evaluation** after each change to see impact:
   ```bash
   python scripts/evaluate_clustering.py
   ```
4. **Compare metrics** like Silhouette Score to find the best weights for your use case

## Understanding Results

After changing weights, check the clustering results:
- **Higher Silhouette Score** = Better-defined clusters
- **Lower Davies-Bouldin Index** = Better separation between clusters
- **Visual inspection** = Look at the UI to see if groupings make sense

## Reverting to Default
Simply reset to:
```python
feature_group_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
```

## Technical Details
- Weights are applied **before** clustering
- Each feature group is internally normalized by `sqrt(group_size)` to balance dimensions
- The multipliers you set are applied **on top** of this normalization
- All three algorithms (K-Means, HDBSCAN, GMM) use the same weights for consistency
