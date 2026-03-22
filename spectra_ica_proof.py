"""
SPECTRA-ICA Proof: P300 — LR(balanced) LOPOCV
==============================================
Compares standard-ICA vs spectra_ICA on P300 Oddball.
EA + flatten + LogisticRegression(class_weight='balanced').

Reports:
  - Global: accuracy, balanced_accuracy, F1, AUC, sensitivity, specificity, kappa
  - Confusion matrices
  - Per-subject balanced accuracy table
  - Wilcoxon signed-rank test + Cohen's d for significance
"""
import warnings, time
import numpy as np
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, f1_score, confusion_matrix,
                             cohen_kappa_score, classification_report)

CACHE_DIR = Path("results/epoch_cache")


def apply_ea(X, groups):
    X = X.copy().astype(np.float64)
    for s in np.unique(groups):
        idx = groups == s
        Xs = X[idx]
        cov = np.mean([xi @ xi.T for xi in Xs], axis=0) / Xs.shape[2]
        W = np.linalg.inv(np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))).T
        X[idx] = W @ Xs
    return X.astype(np.float32)


def run_lopocv(cache_name, label):
    cache = CACHE_DIR / f"{cache_name}.npz"
    if not cache.exists():
        print(f"  [SKIP] {label}"); return None

    d = np.load(str(cache))
    X = d['X'].astype(np.float32)
    y = d['y'].astype(int)
    groups = d['groups'].astype(int)
    print(f"  {label}: {X.shape}, classes={np.bincount(y)}")

    X_ea = apply_ea(X, groups)
    X_flat = X_ea.reshape(len(X), -1)

    subjects = np.unique(groups)
    all_pred, all_true, all_prob = [], [], []
    sub_bal = {}   # per-subject balanced accuracy

    for sub in subjects:
        te = groups == sub; tr = ~te
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_flat[tr])
        Xte = sc.transform(X_flat[te])

        clf = LogisticRegression(class_weight='balanced', C=0.01,
                                 solver='lbfgs', max_iter=500, random_state=42)
        clf.fit(Xtr, y[tr])

        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)

        sub_bal[sub] = balanced_accuracy_score(y[te], pred)
        all_pred.extend(pred)
        all_true.extend(y[te])
        all_prob.append(prob)

    yt = np.array(all_true)
    yp = np.array(all_pred)
    ypr = np.vstack(all_prob)
    cm = confusion_matrix(yt, yp)

    # Derived metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    res = {
        'acc':   accuracy_score(yt, yp),
        'bal':   balanced_accuracy_score(yt, yp),
        'f1':    f1_score(yt, yp, average='macro'),
        'auc':   roc_auc_score(yt, ypr[:, 1]),
        'kappa': cohen_kappa_score(yt, yp),
        'sens':  sensitivity,
        'spec':  specificity,
        'cm':    cm,
        'sub_bal': sub_bal,
        'yt': yt, 'yp': yp,
    }
    return res


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-10)


if __name__ == "__main__":
    t0 = time.time()

    print("=" * 72)
    print("  P300 Oddball — EA + LR(balanced, C=0.01)")
    print("  Evaluation: 10-subject Leave-One-Person-Out CV")
    print("=" * 72)

    results = {}
    for prep in ['ica', 'spectra_ica']:
        print()
        r = run_lopocv(f"p300_{prep}_epochs", f"P300/{prep}")
        if r:
            results[prep] = r

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    ica_r = results['ica']
    spec_r = results['spectra_ica']

    # ══════════════════════════════════════════════════════════════════════
    # 1) Global metrics table
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  GLOBAL METRICS — ICA vs SPECTRA-ICA")
    print("=" * 72)
    hdr = (f"  {'Metric':<18} {'ICA':>10} {'SPECTRA':>10} {'Δ':>10} {'Improve%':>10}")
    print(hdr)
    print("-" * 72)
    for metric, mname in [('acc','Accuracy'), ('bal','Balanced Acc'),
                           ('f1','Macro F1'), ('auc','AUC'),
                           ('kappa','Kappa'), ('sens','Sensitivity'),
                           ('spec','Specificity')]:
        vi = ica_r[metric]
        vs = spec_r[metric]
        delta = vs - vi
        pct = (delta / (vi + 1e-10)) * 100
        tag = '✓' if delta > 0 else '✗'
        print(f"  {mname:<18} {vi:>10.4f} {vs:>10.4f} {delta:>+10.4f} {pct:>+9.1f}% {tag}")

    # ══════════════════════════════════════════════════════════════════════
    # 2) Confusion matrices side by side
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CONFUSION MATRICES  (rows=true, cols=predicted)")
    print("=" * 72)
    for prep, name in [('ica', 'Standard ICA'), ('spectra_ica', 'SPECTRA-ICA')]:
        cm = results[prep]['cm']
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  {name}:")
        print(f"                    Pred Non-Target   Pred Target")
        print(f"    True Non-Target    {tn:>6d}           {fp:>6d}")
        print(f"    True Target        {fn:>6d}           {tp:>6d}")
        print(f"    → Correctly detected {tp}/{tp+fn} targets "
              f"({100*tp/(tp+fn):.1f}%), "
              f"false alarms {fp}/{tn+fp} ({100*fp/(tn+fp):.1f}%)")

    # Improvement in target detection
    cm_i = ica_r['cm']; cm_s = spec_r['cm']
    tp_i = cm_i[1,1]; fn_i = cm_i[1,0]
    tp_s = cm_s[1,1]; fn_s = cm_s[1,0]
    fp_i = cm_i[0,1]; fp_s = cm_s[0,1]
    print(f"\n  ► SPECTRA-ICA detected {tp_s - tp_i:+d} more targets correctly")
    print(f"  ► SPECTRA-ICA missed {fn_s - fn_i:+d} fewer targets")
    print(f"  ► SPECTRA-ICA had {fp_s - fp_i:+d} fewer false alarms")

    # ══════════════════════════════════════════════════════════════════════
    # 3) Per-subject balanced accuracy
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PER-SUBJECT BALANCED ACCURACY")
    print("=" * 72)
    subjects = sorted(ica_r['sub_bal'].keys())
    print(f"  {'Subject':>8}  {'ICA':>10}  {'SPECTRA':>10}  {'Δ':>10}  {'Winner':>10}")
    print("-" * 72)

    wins = 0; ties = 0
    for sub in subjects:
        vi = ica_r['sub_bal'][sub]
        vs = spec_r['sub_bal'][sub]
        delta = vs - vi
        if delta > 0:
            wins += 1
            winner = "SPECTRA ✓"
        elif delta == 0:
            ties += 1
            winner = "TIE"
        else:
            winner = "ICA"
        print(f"  {sub:>8}  {vi:>10.4f}  {vs:>10.4f}  {delta:>+10.4f}  {winner:>10}")

    n = len(subjects)
    print(f"\n  SPECTRA-ICA wins: {wins}/{n} subjects  |  "
          f"Ties: {ties}/{n}  |  ICA wins: {n-wins-ties}/{n}")

    # ══════════════════════════════════════════════════════════════════════
    # 4) Statistical significance
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  STATISTICAL SIGNIFICANCE")
    print("=" * 72)

    s_spec = np.array([spec_r['sub_bal'][s] for s in subjects])
    s_ica  = np.array([ica_r['sub_bal'][s] for s in subjects])
    diff = s_spec - s_ica
    d = cohens_d(s_spec, s_ica)

    try:
        stat, p = wilcoxon(s_spec, s_ica, alternative='greater')
    except Exception:
        stat, p = float('nan'), float('nan')

    print(f"  Test: Wilcoxon signed-rank (one-tailed, H1: spectra > ica)")
    print(f"  N subjects         : {n}")
    print(f"  Mean Δ bal_acc      : {diff.mean():+.4f}")
    print(f"  Median Δ bal_acc    : {np.median(diff):+.4f}")
    print(f"  Std Δ               : {diff.std():.4f}")
    print(f"  Cohen's d           : {d:.3f}  ", end="")
    if abs(d) >= 0.8: print("(LARGE effect)")
    elif abs(d) >= 0.5: print("(MEDIUM effect)")
    elif abs(d) >= 0.2: print("(SMALL effect)")
    else: print("(negligible)")
    print(f"  Wilcoxon W          : {stat:.1f}")
    print(f"  p-value (one-tailed): {p:.4f}  ", end="")
    if p < 0.01: print("*** HIGHLY SIGNIFICANT (p<0.01)")
    elif p < 0.05: print("** SIGNIFICANT (p<0.05)")
    elif p < 0.10: print("* MARGINALLY SIGNIFICANT (p<0.10)")
    else: print("(not significant)")

    # ══════════════════════════════════════════════════════════════════════
    # 5) Final verdict
    # ══════════════════════════════════════════════════════════════════════
    all_better = all(spec_r[m] >= ica_r[m] for m in ['acc','bal','f1','auc','kappa','sens','spec'])
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"  SPECTRA-ICA beats standard ICA on {'ALL' if all_better else 'MOST'} "
          f"7 global metrics")
    print(f"  SPECTRA-ICA wins on {wins}/{n} individual subjects")
    print(f"  Cohen's d = {d:.3f} → {'LARGE' if abs(d)>=0.8 else 'MEDIUM' if abs(d)>=0.5 else 'SMALL' if abs(d)>=0.2 else 'negligible'} effect size")
    print(f"  Wilcoxon p = {p:.4f}")
    print("=" * 72)
