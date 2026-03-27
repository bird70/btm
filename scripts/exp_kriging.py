"""Quick indicator kriging vs KNN spatial CV experiment."""
import warnings
import time

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from pykrige.ok import OrdinaryKriging

warnings.filterwarnings("ignore")

train = pd.read_csv("data/train.csv")
if "ID" not in train.columns:
    train.insert(0, "ID", range(1, len(train) + 1))

enc = LabelEncoder()
y = enc.fit_transform(train["class"])
n_classes = len(enc.classes_)
coords = train[["x", "y"]].values
xs, ys = coords[:, 0], coords[:, 1]

km = KMeans(n_clusters=10, random_state=42, n_init=10)
blocks = km.fit_predict(coords)

max_tr = 3000
oof_kriging = np.zeros((len(y), n_classes))
oof_knn = np.zeros((len(y), n_classes))

t_total = time.time()
for b in range(10):
    va = np.where(blocks == b)[0]
    tr = np.where(blocks != b)[0]

    rng = np.random.RandomState(42)
    tr_sub = rng.choice(tr, min(max_tr, len(tr)), replace=False) if len(tr) > max_tr else tr

    t0 = time.time()
    for c in range(n_classes):
        indicator = (y[tr_sub] == c).astype(float)
        # Skip kriging if class is absent or nearly constant in this fold
        if indicator.sum() < 5 or indicator.sum() > len(indicator) - 5:
            oof_kriging[va, c] = indicator.mean()
            continue
        try:
            ok = OrdinaryKriging(
                xs[tr_sub], ys[tr_sub], indicator,
                variogram_model="exponential",
                verbose=False, enable_plotting=False,
                nlags=20,
            )
            z, _ = ok.execute("points", xs[va], ys[va])
            oof_kriging[va, c] = np.clip(z, 0, 1)
        except (ValueError, LinAlgError):
            # Fallback: use class prevalence
            oof_kriging[va, c] = indicator.mean()

    knn = KNeighborsClassifier(5, weights="distance")
    knn.fit(coords[tr], y[tr])
    oof_knn[va] = knn.predict_proba(coords[va])

    kr_f1 = f1_score(y[va], np.argmax(oof_kriging[va], axis=1), average="weighted")
    kn_f1 = f1_score(y[va], np.argmax(oof_knn[va], axis=1), average="weighted")
    print(f"Block {b} ({len(va)} pts, {len(tr_sub)} tr): kriging={kr_f1:.4f}  knn={kn_f1:.4f}  ({time.time()-t0:.1f}s)")

kr_preds = enc.classes_[np.argmax(oof_kriging, axis=1)]
knn_preds = enc.classes_[np.argmax(oof_knn, axis=1)]
kr_f1_all = f1_score(enc.inverse_transform(y), kr_preds, average="weighted")
knn_f1_all = f1_score(enc.inverse_transform(y), knn_preds, average="weighted")
print()
print(f"Indicator Kriging OOF: {kr_f1_all:.4f}")
print(f"KNN(5) OOF:            {knn_f1_all:.4f}")
print()

for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w * oof_kriging + (1 - w) * oof_knn
    bp = enc.classes_[np.argmax(blend, axis=1)]
    f1 = f1_score(enc.inverse_transform(y), bp, average="weighted")
    print(f"Blend kriging({w:.0%})+knn({1-w:.0%}): {f1:.4f}")

print(f"\nTotal time: {time.time()-t_total:.0f}s")
print("\nKriging classification report:")
print(classification_report(enc.inverse_transform(y), kr_preds))
