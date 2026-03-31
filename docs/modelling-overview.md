# Modelling Overview: Benthic Habitat Classification

A concise guide to the progression of modelling strategies applied in this project, from rule-based terrain classification through to combined multi-scale ML ensembles. All experiments use the same Refugio Cove MBES dataset (6256 training points, 98 test points, 5 classes: ALG, FMAT, NVB, SGAM, SGZ) and report weighted F1 under 5-fold spatial-blocked cross-validation (unless noted).

---

## 1. Baseline: Rule-Based Terrain Classification

**Category: Purely topographic, deterministic**

The BTM library computes standard terrain derivatives from the bathymetry raster — BPI (Bathymetric Position Index), Slope, VRM (Vector Ruggedness Measure), Surface Ratio — then assigns habitat classes using a user-defined lookup table (depth × BPI zones). No training data is required. This is the classical benthic terrain method (Wright & Lundblad 2005) and the starting point for all subsequent ML work.

---

## 2. MBES Core Features + Random Forest

**Category: Single ML, point-sample features**

The `benthic_model` pipeline extracts 8 point-sample features directly from the bathymetry and backscatter rasters at each labelled location: depth, backscatter, slope, VRM, complexity, max-curvature, northness, eastness. A Random Forest classifier is trained on these features with class-balanced bootstrapping. This is the **reference baseline** (R04/R06, run-014): **CV F1 = 0.8024, Kaggle F1 = 0.79518**. All subsequent CV targets are expressed relative to this result.

---

## 3. BTM Terrain Derivatives as ML Features

**Category: Single ML, terrain-engineered features**

BTM derivatives (fine-scale BPI, broad-scale BPI, slope, surface ratio, VRM) are added _alongside_ the MBES-8 features and fed into Random Forest. Adding BTM terrain features closed most of the CV–Kaggle gap vs earlier approaches and matched the reference baseline (R04/R06). RF with BTM features generalised better than gradient-boosted models (LGB, CatBoost, XGBoost), which all scored lower on the same feature set (run-014, Phase 3). Key finding: `class_weight='balanced_subsample'` on RF handles the 5-class imbalance better than boosted alternatives.

---

## 4. Object-Based Image Analysis (OBIA) + Pixel Hybrid

**Category: Hybrid ML, segmentation-assisted**

Following Ierodiaconou et al. (2018), SLIC superpixel segmentation groups spatially adjacent pixels into objects using bathymetry + backscatter + VRM. Per-object summary statistics (mean, std) replace raw pixel values as ML features. A soft-vote ensemble of the pixel-based (PB) and object-based (OB) CatBoost predictions is combined. The OBIA pipeline achieved **CV F1 ≈ 0.576** on pixel features alone, well below the MBES-8 RF baseline, and was not pursued further (run-009).

---

## 5. Model Comparison Sweep

**Category: Single ML, model selection**

Run-014 benchmarked RF, LightGBM, CatBoost, and XGBoost on the same BTM-winner feature set under identical spatial-blocked CV. Result: RF remained best (0.8024), followed by CatBoost (0.797), LGB (0.795), Ensemble RF+LGB (0.797), XGBoost (0.775). CatBoost GPU produced the highest CV score seen (0.8139) but the largest CV–Kaggle gap (0.052), suggesting overfitting to spatial patterns not present in the test partition.

---

## 6. Ecology-Informed Features

**Category: Single ML, domain-guided feature engineering**

Ecology-derived indicators were added to the BTM feature set to better separate the rare SGAM class: binary depth-zone flags (shallow / mid / deep), a SGAM niche indicator (shallow + low BPI + low slope), and raster-sampled eco-derivatives (northness, eastness, max-curvature, complexity). The SGAM CV recall improved from 0.043 → 0.045, meeting the target gate, but overall Kaggle F1 did not improve. Raster eco-derivatives contributed no signal beyond depth zones alone (run-015).

---

## 7. Multi-Scale Terrain + GLCM Texture (BTM-Only)

**Category: Single ML, multi-scale feature engineering**

Rather than single-scale BTM derivatives, this approach computed terrain features across 5 spatial window sizes (3, 7, 11, 15, 21 cells): slope, VRM, surface ratio, northness, eastness, max-curvature, complexity, RDMV (depth variability), plus GLCM backscatter texture (contrast, homogeneity) — 64 features in total, reduced to 33 via Spearman correlation pre-filter and permutation importance. These were BTM-only standalone features (MBES-8 not included). Best CV F1 = **0.6383** (CatBoost), a +11 pp gain over BTM-base CatBoost (~0.52) but far below the MBES-8 reference (0.8024) because depth/backscatter were absent (run-016).

---

## 8. Combined BTM-33 + MBES-8 (Current Best Standalone)

**Category: Ensemble ML, combined terrain + acoustic point-sample**

The 33 selected multi-scale BTM features (from run-016) were combined with the 8 MBES point-sample features, giving 41 features in total. Three models were trained and evaluated — RF, CatBoost, LightGBM — plus a 3-model soft-vote ensemble. Feature selection (permutation importance) reduced to 38 features with no CV degradation. CatBoost with 38 features achieved the best result: **CV F1 = 0.7829**, a +14.5 pp gain over BTM-only. RF CV F1 = 0.7712, still 3.1 pp below the reference baseline, suggesting the 33 BTM features add noise relative to the focused 10-feature set in R04. SGAM recall improved dramatically: 0.142 vs 0.000 in run-016, driven by the MBES-8 features and the 5-fold spatial-blocked CV scheme (run-017).

---

## 9. Spatial Cross-Validation Strategy

**Category: Evaluation methodology (applies to all ML runs)**

Early experiments used 10-fold KMeans GroupKFold spatial blocking. From run-014 onwards, 5-fold spatial-blocked CV (`iter_spatial_blocked_folds`, 4×4 quantile bins) was adopted to match the reference R04/R06 results. The 5-fold scheme with 4 spatial bins leads to larger training folds and better distributes rare-class samples (especially SGAM) across folds. Adding spatial coordinates (x, y) as features did not inflate CV score, confirming the blocking correctly prevents spatial memorisation.

---

## 10. Summary: CV F1 Progression

| Run               | Approach                      | Features | Model        | CV F1      | Kaggle F1   |
| ----------------- | ----------------------------- | -------- | ------------ | ---------- | ----------- |
| R04/R06           | MBES-8 + BTM (reference)      | ~10      | RF           | **0.8024** | **0.79518** |
| R09 (run-015)     | BTM + CatBoost GPU            | ~10      | CatBoost GPU | 0.8139     | 0.76153     |
| R15/R16 (run-015) | BTM + eco features            | ~12–16   | RF           | 0.7916     | 0.76438     |
| run-016           | Multi-scale BTM-33 (BTM-only) | 33       | CatBoost     | 0.6383     | —           |
| run-017           | BTM-33 + MBES-8 combined      | 38–41    | CatBoost     | 0.7829     | —           |
| OBIA (run-009)    | Pixel+object hybrid           | 21       | CatBoost     | 0.5758     | 0.76394     |

**Key insight**: The MBES-8 point-sample features (especially depth) carry ~14 pp of predictive signal. Multi-scale BTM features add texture and terrain context but are additive on top of — not a replacement for — direct acoustic measurements. The current best standalone result (run-017, CatBoost 0.7829) is 3.1 pp below the R04 reference RF, likely because the large BTM feature set dilutes the RF's signal concentration advantage.
