import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from scipy import stats
from itertools import combinations
from preprocessing import preprocess_subject
from config import CLASS_NAMES

OZ_CH = 4

print("="*60)
print("Oz Electrode Confound Analysis")
print("H0: No significant difference in Oz activity between MI classes")
print("="*60)

all_results = {}

for subj in range(1, 10):
    print(f"\nSubject {subj:02d}:")
    X, y = preprocess_subject(subj, session='T', verbose=True)
    oz_data = X[:, :, :, OZ_CH]
    oz_power = oz_data.mean(axis=(1, 2))

    class_power = {cls: oz_power[y == cls] for cls in range(4)}

    pairs = list(combinations(range(4), 2))
    pair_results = {}
    for a, b in pairs:
        t, p = stats.ttest_ind(class_power[a], class_power[b])
        pair_results[(a, b)] = p

    sig = {(CLASS_NAMES[a], CLASS_NAMES[b]): round(p, 4)
           for (a, b), p in pair_results.items() if p < 0.05}
    nonsig = {(CLASS_NAMES[a], CLASS_NAMES[b]): round(p, 4)
              for (a, b), p in pair_results.items() if p >= 0.05}

    all_results[subj] = {'sig': sig, 'nonsig': nonsig, 'all': pair_results}

    if sig:
        print(f"  Significant pairs (p<0.05): {sig}")
    else:
        print(f"  No significant class differences on Oz (all p>0.05)")
    print(f"  Non-significant: {nonsig}")

print("\n" + "="*60)
print("SUMMARY: Oz significant pairs per subject")
print("="*60)
print(f"{'Subj':<6} {'Sig/6':<8} {'Verdict'}")
print("-"*40)
for subj, res in all_results.items():
    n_sig = len(res['sig'])
    verdict = "CONFOUND RISK" if n_sig > 0 else "CLEAN"
    print(f"S{subj:02d}   {n_sig}/6     {verdict}")
    if n_sig > 0:
        for pair, p in res['sig'].items():
            print(f"       {pair[0]} vs {pair[1]}: p={p}")

n_confound = sum(1 for r in all_results.values() if len(r['sig']) > 0)
print(f"\nSubjects with ≥1 significant Oz pair: {n_confound}/9")

if n_confound <= 3:
    print("→ CONCLUSION: Oz confound is limited to a minority of subjects.")
    print("  Add as limitation; no architectural change needed.")
elif n_confound <= 6:
    print("→ CONCLUSION: Oz shows moderate confound. Recommend acknowledging")
    print("  in limitations and reporting results with/without Oz channel.")
else:
    print("→ CONCLUSION: Oz is systematically confounded. Consider removing it")
    print("  and re-running, or reporting a sensitivity analysis without Oz.")