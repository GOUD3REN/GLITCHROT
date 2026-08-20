import argparse
import os
import yaml
import joblib
import numpy as np
import tqdm
from pathlib import Path

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------
# IMPORTS FROM FEATURES.PY
# --------------------------------------------------------------
from features import (               # <-- ALL FUNCTIONS ARE NOW AVAILABLE
    get_embeddings_from_folder,
    extract_metadata_features,
    close_hooks,
)

# ----------------------------------------------------------------------
#CONFIG LOADER
# ----------------------------------------------------------------------
def load_cfg(cfg_path: Path) -> dict:
    """Load a YAML config file using UTF-8 (fixes accent issues)."""

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----------------------------------------------------------------------
#FIND SUB‑FOLDER (case‑insensitive)
# ----------------------------------------------------------------------
def _find_subfolder(root: Path, name_lower: str) -> Path:
    """Return the sub‑directory whose name matches name_lower (case‑insensitive)."""

    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name_lower:
            return p

    raise FileNotFoundError(
        f'Could not find a sub-directory named "{name_lower}" under {root}.'
    )

# ----------------------------------------------------------------------
#     RUN OPTUNA (hyper‑parameter search)
# --------------------------------------------------------------
def run_optuna(cfg: dict, emb_real, paths_real, emb_fake, paths_fake):
    # ------------------------------------------------------------------
    # Determine the minimum number of samples between the two classes
    # ------------------------------------------------------------------
    n_real = len(paths_real)
    n_fake = len(paths_fake)
    min_len = min(n_real, n_fake)

    # Trim both embeddings and path lists to that minimum size
    emb_real = emb_real[:min_len]
    emb_fake = emb_fake[:min_len]
    paths_real = paths_real[:min_len]
    paths_fake = paths_fake[:min_len]

    # ------------------------------------------------------------------
    # Embeddings + labels (balanced classes)
    # ------------------------------------------------------------------
    X_emb = np.concatenate([emb_real, emb_fake], axis=0)
    y = np.array([0] * len(emb_real) + [1] * len(emb_fake))

    # ------------------------------------------------------------------
    # METADATA COLLECTION (EXIF/PRNU) – always 2‑D
    # ------------------------------------------------------------------
    meta_features = []
    for p in paths_real:
        meta_features.append(extract_metadata_features(p))
    for p in paths_fake:
        meta_features.append(extract_metadata_features(p))

    # Stack metadata; shape will be (2*min_len, 1)
    meta_arr = np.stack(meta_features, axis=0)

    # ------------------------------------------------------------------
    # Flatten embeddings to 2‑D (required for hstack later)
    # ------------------------------------------------------------------
    # If any dimension beyond the first is present (e.g. 4‑D image embeddings),
    # collapse them into a 2‑D matrix of shape (samples, features).
    if X_emb.ndim > 2:
        X_emb = X_emb.reshape(len(X_emb), -1)   # (samples, flat_features)

    # Ensure meta_arr is 2‑D
    if meta_arr.ndim != 2:
        meta_arr = meta_arr.reshape(-1, 1)

    # Final sanity check that row counts match before proceeding
    assert X_emb.shape[0] == meta_arr.shape[0], (
        f"After trimming, sample count mismatch – X_emb rows={X_emb.shape[0]}, "
        f"meta_arr rows={meta_arr.shape[0]}"
    )

    # ------------------------------------------------------------------
    # OPTUNA SEARCH
    # ------------------------------------------------------------------
    import optuna

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                cfg["search_spaces"]["n_estimators"]["min"],
                cfg["search_spaces"]["n_estimators"]["max"],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                cfg["search_spaces"]["max_depth"]["min"],
                cfg["search_spaces"]["max_depth"]["max"],
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                cfg["search_spaces"]["min_samples_leaf"]["min"],
                cfg["search_spaces"]["min_samples_leaf"]["max"],
            ),
            "bootstrap": trial.suggest_categorical(
                "bootstrap",
                cfg["search_spaces"]["bootstrap"]["choices"]
            ),
            "class_weight": "balanced",
            "random_state": cfg["random_state"],
        }

        # ---------- TRAIN/VALIDATION SPLIT ----------
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_emb, y,
            test_size=cfg["train_test_split_ratio"],
            stratify=y,
            random_state=cfg["random_state"]
        )

        # ---------- CALIBRATED CLASSIFIER ----------
        # NOTE: cv is forced to 2 (or another safe value) when the split
        # is tiny.  You can also switch cfg["calibration_method"] to
        # "sigmoid" or "platt" if you prefer a CV‑free method.
        cal = CalibratedClassifierCV(
            RandomForestClassifier(**params),
            method=cfg["calibration_method"],
            cv=2   # reduced from 5 to a safe default
        )
        cal.fit(X_tr, y_tr)

        # ---------- VALIDATION METRICS ----------
        prob = cal.predict_proba(X_val)[:, 1]

        thr = np.linspace(0, 1, 200)

        f1s = [
            f1_score(y_val, (prob >= t).astype(int))
            for t in thr
        ]

        return -max(f1s)          # Optuna minimizes → negative F1

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.get("n_trials", 30))

    best = study.best_params
    best["class_weight"] = "balanced"
    best["random_state"] = cfg["random_state"]
    return best

# ----------------------------------------------------------------------
#FINAL MODEL TRAINING
# ----------------------------------------------------------------------
def train_final(cfg: dict, best_params: dict, X, y):
    """Train the final calibrated model and persist artifacts."""

    #Train/validation split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg["train_test_split_ratio"],
        stratify=y,
        random_state=cfg["random_state"]
    )

    #Model + calibration
    clf = RandomForestClassifier(**best_params)

    cal = CalibratedClassifierCV(
        clf,
        method=cfg["calibration_method"],
        cv=2   # keep the same safe CV value used in `run_optuna`
    )
    cal.fit(X_tr, y_tr)

    #Validation metrics + optimal threshold
    prob = cal.predict_proba(X_te)[:, 1]

    thr = np.linspace(0, 1, 200)

    f1s = [
        f1_score(y_te, (prob >= t).astype(int))
        for t in thr
    ]

    best_thr = thr[np.argmax(f1s)]

    pred = (prob >= best_thr).astype(int)

    print("\n=== VALIDATION METRICS ===")
    print(f"Accuracy : {accuracy_score(y_te, pred):.4f}")
    print(f"F1-Score : {f1_score(y_te, pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_te, prob):.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_te,
            pred,
            target_names=["Real", "Fake"]
        )
    )

    #Persist artefacts
    joblib.dump(cal, cfg["model_output_path"])
    np.save(cfg["best_threshold_path"], best_thr)
    joblib.dump(best_params, cfg["best_params_path"])

    # Tiny summary yaml
    cfg["best_accuracy"] = float(accuracy_score(y_te, pred))
    cfg["best_f1"] = float(f1_score(y_te, pred))
    cfg["best_auc"] = float(roc_auc_score(y_te, prob))
    cfg["best_threshold"] = float(best_thr)

    open(
        "experiment_summary.yaml",
        "w"
    ).write(yaml.safe_dump(cfg))

    return cal

# ----------------------------------------------------------------------
#     MAIN – PARSE ARGS, LOAD CFG, RUN PIPELINE
# ----------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml")
    )

    parser.add_argument(
        "--dataset_root",
        type=Path,
        help="Optional override for the dataset root path (useful on Windows)",
    )

    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Override dataset_root if a CLI flag was supplied
    if args.dataset_root is not None:
        cfg["dataset_root"] = str(args.dataset_root)

    # --------------------------------------------------------------
    #   Locate the real/ fake sub‑folders (case‑insensitive)
    # ------------------------------------------------------------------
    root = Path(cfg["dataset_root"])

    # Helper that finds a sub‑folder by name (case‑insensitive)
    root = Path(cfg["dataset_root"])

    real_dir = _find_subfolder(root, "real")
    fake_dir = _find_subfolder(root, "fake")

    # --------------------------------------------------------------
    #   Load embeddings
    # --------------------------------------------------------------
    real_emb, real_paths = get_embeddings_from_folder(real_dir)
    fake_emb, fake_paths = get_embeddings_from_folder(fake_dir)

    # --------------------------------------------------------------
    #   Hyper‑parameter optimisation
    # --------------------------------------------------------------
    best_params = run_optuna(
        cfg,
        real_emb,
        real_paths,
        fake_emb,
        fake_paths
    )

    # --------------------------------------------------------------
    #   Build final feature matrix (embeddings + metadata)
    # --------------------------------------------------------------
    # Trim path lists to the smaller size so both classes match
    n_real = len(real_paths)
    n_fake = len(fake_paths)
    min_len = min(n_real, n_fake)

    # Trim the path lists
    real_paths = real_paths[:min_len]
    fake_paths = fake_paths[:min_len]

    # Collect metadata only for the trimmed paths
    meta_features = []
    for p in real_paths:
        meta_features.append(extract_metadata_features(p))
    for p in fake_paths:
        meta_features.append(extract_metadata_features(p))
    meta_arr = np.stack(meta_features, axis=0)          # (2*min_len, 1)

    # Flatten embeddings to 2‑D and keep only the first `min_len` rows per class
    # (already trimmed above, but we ensure the shape is flat)
    real_emb_flat = real_emb.reshape(real_emb.shape[0], -1)[:min_len]
    fake_emb_flat = fake_emb.reshape(fake_emb.shape[0], -1)[:min_len]

    # Build a *single* matrix that contains **both** real and fake samples.
    # Each sample gets its own metadata row that is already stacked in `meta_arr`.
    emb_total = np.concatenate([real_emb_flat, fake_emb_flat], axis=0)   # (2*min_len, embed_dim)
    # Safety: make sure meta_arr is 2‑D
    if meta_arr.ndim != 2:
        meta_arr = meta_arr.reshape(-1, 1)
    # Final feature matrix: embedding vector + metadata (shape → (2*min_len, embed_dim+1))
    X = np.hstack([emb_total, meta_arr])                # (2*min_len, embed_dim+1)
    # Corresponding label vector: 0 for real, 1 for fake (length = 2*min_len)
    y = np.array([0] * min_len + [1] * min_len)

    # --------------------------------------------------------------
    #   Train final model
    # --------------------------------------------------------------
    final_model = train_final(cfg, best_params, X, y)

    # ------------------------------------------------------------------
    #  Clean up
    # ----------------------------------------------------------------------
    close_hooks()
    print(
        "\n[INFO] All done – model saved to:",
        cfg["model_output_path"]
    )
