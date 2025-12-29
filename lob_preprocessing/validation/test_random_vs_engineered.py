"""
Statistical test: Random vs Engineered features

Question: Is engineered significantly better than random?
"""

import numpy as np
from scipy import stats

# Results from experiments
raw_accs = np.array([62.05, 62.18, 62.87, 63.12, 62.82]) / 100
random_accs = np.array([65.13, 64.84, 65.44, 64.88, 64.83]) / 100
eng_accs = np.array([68.87, 68.95, 68.82, 68.91, 68.96]) / 100

print("="*60)
print("STATISTICAL COMPARISON: RANDOM VS ENGINEERED")
print("="*60)

# Test 1: Raw vs Random
print("\n1. Raw vs Raw+Random:")
t_stat, p_val = stats.ttest_rel(raw_accs, random_accs)
print(f"   Mean diff: {(random_accs.mean() - raw_accs.mean())*100:.2f} pp")
print(f"   t-statistic: {t_stat:.2f}")
print(f"   p-value: {p_val:.6f}")
if p_val < 0.05:
    print("   ✅ Significant (p < 0.05)")
else:
    print("   ❌ Not significant (p >= 0.05)")

# Test 2: Random vs Engineered
print("\n2. Raw+Random vs Raw+Engineered:")
t_stat, p_val = stats.ttest_rel(random_accs, eng_accs)
print(f"   Mean diff: {(eng_accs.mean() - random_accs.mean())*100:.2f} pp")
print(f"   t-statistic: {t_stat:.2f}")
print(f"   p-value: {p_val:.6f}")
if p_val < 0.05:
    print("   ✅ Significant (p < 0.05)")
else:
    print("   ❌ Not significant (p >= 0.05)")

# Test 3: Raw vs Engineered (already known)
print("\n3. Raw vs Raw+Engineered:")
t_stat, p_val = stats.ttest_rel(raw_accs, eng_accs)
print(f"   Mean diff: {(eng_accs.mean() - raw_accs.mean())*100:.2f} pp")
print(f"   t-statistic: {t_stat:.2f}")
print(f"   p-value: {p_val:.6f}")
if p_val < 0.05:
    print("   ✅ Significant (p < 0.05)")
else:
    print("   ❌ Not significant (p >= 0.05)")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print()
print("Feature dimensionality effect:")
print(f"  Raw (40) → Raw+Random (78): +{(random_accs.mean() - raw_accs.mean())*100:.2f} pp")
print()
print("Domain knowledge effect:")
print(f"  Raw+Random (78) → Raw+Eng (78): +{(eng_accs.mean() - random_accs.mean())*100:.2f} pp")
print()
print("Total effect:")
print(f"  Raw (40) → Raw+Eng (78): +{(eng_accs.mean() - raw_accs.mean())*100:.2f} pp")
print()
print("Decomposition:")
print(f"  Total = Dimensionality + Domain knowledge")
print(f"  {(eng_accs.mean() - raw_accs.mean())*100:.2f} = {(random_accs.mean() - raw_accs.mean())*100:.2f} + {(eng_accs.mean() - random_accs.mean())*100:.2f} pp")
print()
print("✅ This shows that BOTH contribute, but domain knowledge")
print("   provides substantially more benefit (+3.87 pp vs +2.42 pp)")
print()
print("="*60)
