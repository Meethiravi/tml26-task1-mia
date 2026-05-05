import os
import sys
import requests
import time
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from scipy.stats import norm

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "YOUR_API_KEY_HERE" #REPLACE WITH YOUR API KEY
TASK_ID = "01-mia"  #DONOT CHANGE

# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"
SAFE_CSV = BASE / "submission_safe.csv"

CKPT_DIR = BASE
SPLITS_NPZ = CKPT_DIR / "lira_v7_splits.npz"
FEATURES_NPZ = CKPT_DIR / "lira_v7_features.npz"

NUM_CLASSES = 9
NUM_TOTAL_SHADOWS = 60
REF_EPOCHS = 30
REF_BATCH_SIZE = 128
REF_LR = 0.1
REF_WEIGHT_DECAY = 5e-4
SEED = 42

# normalization
MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

# device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform
    def __getitem__(self, i):
        id_, img, label = self.ids[i], self.imgs[i], self.labels[i]
        if self.transform: img = self.transform(img)
        return id_, img, label
    def __len__(self): return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []
    def __getitem__(self, i):
        id_, img, label = super().__getitem__(i)
        return id_, img, label, self.membership[i]
    
class TrainSubset(Dataset):
    def __init__(self, base_ds, indices):
        self.base = base_ds; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        real_i = self.indices[i]
        img = self.base.imgs[real_i]
        label = self.base.labels[real_i]
        if img.dim() == 3 and img.shape[1] != 32:
            img = TF.resize(img, [32, 32])
        img = TF.normalize(img, MEAN, STD)
        return img, label

if __name__ == "__main__":

    def make_resnet18():
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, NUM_CLASSES)
        return model

    print("Loading datasets...")
    pub_ds = torch.load(PUB_PATH, weights_only=False)
    priv_ds = torch.load(PRIV_PATH, weights_only=False)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    pub_ds.transform = transform
    priv_ds.transform = transform
    priv_ds.membership = [-1] * len(priv_ds.ids)

    print("Loading target model...")
    target_model = make_resnet18()
    target_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    target_model.to(device).eval()

    n_pub = len(pub_ds)
    n_priv = len(priv_ds)
    print(f"Pub: {n_pub}, Priv: {n_priv}")


    v4_splits_path = CKPT_DIR / "lira_v4_splits.npz"
    v5_splits_path = CKPT_DIR / "lira_v5_splits.npz"

    if SPLITS_NPZ.exists():
        print(f"Loading cached splits from {SPLITS_NPZ}")
        in_mask = np.load(SPLITS_NPZ)["in_mask"]
    elif v5_splits_path.exists():
        v5_in_mask = np.load(v5_splits_path)["in_mask"]
        rng = np.random.RandomState(SEED + 7777)
        new_count = NUM_TOTAL_SHADOWS - v5_in_mask.shape[0]
        new_masks = np.zeros((new_count, n_pub), dtype=bool)
        for r in range(new_count):
            perm = rng.permutation(n_pub)
            new_masks[r, perm[:n_pub // 2]] = True
        in_mask = np.vstack([v5_in_mask, new_masks])
        np.savez(SPLITS_NPZ, in_mask=in_mask)
        print(f"Saved combined splits to {SPLITS_NPZ}")
    elif v4_splits_path.exists():
        v4_in_mask = np.load(v4_splits_path)["in_mask"]
        rng = np.random.RandomState(SEED + 7777)
        new_count = NUM_TOTAL_SHADOWS - v4_in_mask.shape[0]
        new_masks = np.zeros((new_count, n_pub), dtype=bool)
        for r in range(new_count):
            perm = rng.permutation(n_pub)
            new_masks[r, perm[:n_pub // 2]] = True
        in_mask = np.vstack([v4_in_mask, new_masks])
        np.savez(SPLITS_NPZ, in_mask=in_mask)
        print(f"Saved combined splits to {SPLITS_NPZ}")
    else:
        print("Generating 32 fresh random splits...")
        rng = np.random.RandomState(SEED)
        in_mask = np.zeros((NUM_TOTAL_SHADOWS, n_pub), dtype=bool)
        for r in range(NUM_TOTAL_SHADOWS):
            perm = rng.permutation(n_pub)
            in_mask[r, perm[:n_pub // 2]] = True
        np.savez(SPLITS_NPZ, in_mask=in_mask)
        print(f"Saved fresh splits to {SPLITS_NPZ}")

    print(f"Splits shape: {in_mask.shape}, each shadow on {in_mask[0].sum()} samples")


    def train_shadow(train_indices, ref_id, seed):
        torch.manual_seed(seed); np.random.seed(seed)

        train_ds = TrainSubset(pub_ds, train_indices)
        loader = DataLoader(train_ds, batch_size=REF_BATCH_SIZE, shuffle=True, num_workers=0)

        model = make_resnet18().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=REF_LR, momentum=0.9,
                                    weight_decay=REF_WEIGHT_DECAY, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=REF_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        print(f"\n--- Training shadow {ref_id} (seed={seed}) on {len(train_indices)} samples ---")
        model.train()
        t0 = time.time()
        for epoch in range(REF_EPOCHS):
            running_loss = 0.0; correct = 0; total = 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                total += imgs.size(0)
            scheduler.step()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"shadow{ref_id} ep{epoch+1}/{REF_EPOCHS}: "
                    f"loss={running_loss/total:.4f} acc={correct/total:.4f} "
                    f"t={time.time()-t0:.0f}s")
        model.eval()
        return model

    # checks if we have existing checkpoints
    def find_existing_checkpoint(ref_id):
        candidates = [
            CKPT_DIR / f"ref_v7_{ref_id}.pt",        
            CKPT_DIR / f"ref_v5_{ref_id}.pt",       
            CKPT_DIR / f"ref_v4_{ref_id}.pt",      
        ]
        for c in candidates:
            if c.exists():
                return c
        return None


    print(f"\n Training/loading shadow models")
    shadow_models = []
    for r in range(NUM_TOTAL_SHADOWS):
        existing = find_existing_checkpoint(r)
        if existing is not None:
            print(f"Loading shadow {r} from {existing.name}")
            m = make_resnet18().to(device)
            m.load_state_dict(torch.load(existing, map_location=device))
            m.eval()
        else:
            train_indices = np.where(in_mask[r])[0]
            m = train_shadow(train_indices, r, seed=SEED + 1000 * (r + 1))
            ckpt_path = CKPT_DIR / f"ref_v7_{r}.pt"
            torch.save(m.state_dict(), ckpt_path)
            print(f"Saved shadow {r} to {ckpt_path.name}")
        shadow_models.append(m)
    print(f"Total shadows: {len(shadow_models)}")

    def _logit_gap(logits, labels):
        correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(logits).scatter_(1, labels.view(-1, 1), 0.)
        others = logits * mask - (1 - mask) * 1e9
        return correct - others.max(1)[0]

    @torch.no_grad()
    def extract_rich_features(model, loader):
        model.eval()
        features = {
            "loss": [], "gap": [],
            "max_conf": [], "entropy": [], "margin": [],
        }
        softmax_all = []
        all_ids, all_labels, all_mem = [], [], []

        for batch in loader:
            if len(batch) == 4:
                ids, imgs, labels, mem = batch
            else:
                ids, imgs, labels = batch; mem = [-1]*len(ids)
            imgs = imgs.to(device); labels = labels.to(device)
            bs = imgs.size(0)

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            sorted_probs, _ = probs.sort(dim=1, descending=True)

            loss = F.cross_entropy(logits, labels, reduction='none')
            gap = _logit_gap(logits, labels)
            max_conf = sorted_probs[:, 0]
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]

            features["loss"].append(loss.cpu().numpy())
            features["gap"].append(gap.cpu().numpy())
            features["max_conf"].append(max_conf.cpu().numpy())
            features["entropy"].append(entropy.cpu().numpy())
            features["margin"].append(margin.cpu().numpy())
            softmax_all.append(probs.cpu().numpy())

            all_ids.extend([int(x) for x in ids])
            all_labels.extend(labels.cpu().numpy().tolist())
            all_mem.extend([int(m.item()) if torch.is_tensor(m) else int(m) for m in mem])

        for k in features:
            features[k] = np.concatenate(features[k])
        softmax_all = np.concatenate(softmax_all, axis=0)
        return (features, softmax_all,
                np.array(all_ids), np.array(all_labels), np.array(all_mem))


    pub_loader = DataLoader(pub_ds, batch_size=128, shuffle=False, num_workers=0)
    priv_loader = DataLoader(priv_ds, batch_size=128, shuffle=False, num_workers=0)


    if FEATURES_NPZ.exists():
        print(f"\nLoading cached features from {FEATURES_NPZ}")
        z = np.load(FEATURES_NPZ)
        target_pub = {k.replace("tpub_", ""): z[k] for k in z.files if k.startswith("tpub_")}
        target_priv = {k.replace("tpriv_", ""): z[k] for k in z.files if k.startswith("tpriv_")}
        target_softmax_pub = z["target_softmax_pub"]
        target_softmax_priv = z["target_softmax_priv"]
        shadow_pub = z["shadow_pub"]
        shadow_priv = z["shadow_priv"]
        shadow_softmax_pub = z["shadow_softmax_pub"]
        shadow_softmax_priv = z["shadow_softmax_priv"]
        pub_mem = z["pub_mem"]; pub_ids = z["pub_ids"]; priv_ids = z["priv_ids"]
        pub_labels = z["pub_labels"]; 
        priv_labels = z["priv_labels"]
        feature_names = list(z["feature_names"])
    else:
        print("\n Extracting features")
        target_pub, target_softmax_pub, pub_ids, pub_labels, pub_mem = extract_rich_features(
            target_model, pub_loader)
        target_priv, target_softmax_priv, priv_ids, priv_labels, _ = extract_rich_features(
            target_model, priv_loader)

        feature_names = list(target_pub.keys())
        n_features = len(feature_names)

        shadow_pub = np.zeros((n_pub, NUM_TOTAL_SHADOWS, n_features))
        shadow_priv = np.zeros((n_priv, NUM_TOTAL_SHADOWS, n_features))
        shadow_softmax_pub = np.zeros((n_pub, NUM_TOTAL_SHADOWS, NUM_CLASSES))
        shadow_softmax_priv = np.zeros((n_priv, NUM_TOTAL_SHADOWS, NUM_CLASSES))

        for r, m in enumerate(shadow_models):

            feats_pub, sm_pub, _, _, _ = extract_rich_features(m, pub_loader)
            for fi, fname in enumerate(feature_names):
                shadow_pub[:, r, fi] = feats_pub[fname]
            shadow_softmax_pub[:, r, :] = sm_pub


            feats_priv, sm_priv, _, _, _ = extract_rich_features(m, priv_loader)
            for fi, fname in enumerate(feature_names):
                shadow_priv[:, r, fi] = feats_priv[fname]
            shadow_softmax_priv[:, r, :] = sm_priv

        save_dict = {
            "shadow_pub": shadow_pub, "shadow_priv": shadow_priv,
            "shadow_softmax_pub": shadow_softmax_pub, "shadow_softmax_priv": shadow_softmax_priv,
            "target_softmax_pub": target_softmax_pub, "target_softmax_priv": target_softmax_priv,
            "pub_mem": pub_mem, "pub_ids": pub_ids, "priv_ids": priv_ids,
            "pub_labels": pub_labels, "priv_labels": priv_labels,
            "feature_names": np.array(feature_names),
        }
        for k, v in target_pub.items():
            save_dict[f"tpub_{k}"] = v
        for k, v in target_priv.items():
            save_dict[f"tpriv_{k}"] = v
        np.savez(FEATURES_NPZ, **save_dict)
        print(f"Saved features to {FEATURES_NPZ}")

    print("\n Building attack features")

    def compute_out_stats(target_vals_dict, shadow_vals, in_mask_, is_priv, fnames):
        n, R, F_ = shadow_vals.shape
        out_features = {}
        for fi, fname in enumerate(fnames):
            target_v = target_vals_dict[fname]
            shadow_v = shadow_vals[:, :, fi]

            if any(x in fname for x in ["loss", "entropy", "var"]):
                target_v = np.log(target_v + 1e-12)
                shadow_v = np.log(shadow_v + 1e-12)

            out_means = np.zeros(n)
            out_stds = np.zeros(n)
            
            for i in range(n):
                if is_priv:
                    out_refs = np.zeros(R, dtype=bool)
                    out_refs[:R//2] = True 
                else:
                    out_refs = ~in_mask_[:, i]
                
                if out_refs.sum() < 2: out_refs = np.ones(R, dtype=bool)
                vals = shadow_v[i, out_refs]
                out_means[i] = vals.mean()
                out_stds[i] = vals.std() if len(vals) > 1 else 1.0

            if any(x in fname for x in ["loss", "entropy", "var"]):

                z_scores = (out_means - target_v) / (out_stds + 1e-6)
                diffs = out_means - target_v

            else:
                z_scores = (target_v - out_means) / (out_stds + 1e-6)
                diffs = target_v - out_means

            out_features[f"{fname}_z"] = z_scores
            out_features[f"{fname}_diff"] = diffs
                
            log_prob_out = norm.logpdf(target_v, loc=out_means, scale=out_stds + 1e-6)
            out_features[f"{fname}_lira"] = -log_prob_out 

        return out_features

    def softmax_features(target_probs, shadow_probs, in_mask_, is_priv):
        n, R, C = shadow_probs.shape

        out_means = np.zeros((n, C))
        in_means = np.zeros((n, C))

        for i in range(n):
            if is_priv:
                out_refs = np.ones(R, dtype=bool)
                in_refs = np.ones(R, dtype=bool)
            else:
                out_refs = ~in_mask_[:, i]
                in_refs = in_mask_[:, i]

                if out_refs.sum() < 2:
                    out_refs = np.ones(R, dtype=bool)
                if in_refs.sum() < 2:
                    in_refs = np.ones(R, dtype=bool)

            out_means[i] = shadow_probs[i, out_refs, :].mean(axis=0)
            in_means[i] = shadow_probs[i, in_refs, :].mean(axis=0)

        diff_out = target_probs - out_means

        in_out_gap = in_means - out_means

        return np.hstack([
            target_probs,
            diff_out,
            in_out_gap
        ])


    print("Computing features loss/gap/margion/max_conf/entropy: v5")
    pub_feats = compute_out_stats(target_pub, shadow_pub, in_mask, is_priv=False, fnames=feature_names)
    priv_feats = compute_out_stats(target_priv, shadow_priv, in_mask, is_priv=True, fnames=feature_names)
    keys = [k for k in pub_feats.keys()]
    stats_pub = np.column_stack([pub_feats[k] for k in keys])
    stats_priv = np.column_stack([priv_feats[k] for k in keys])

    print("Computing softmax features: v6")
    X_pub_sm = softmax_features(target_softmax_pub, shadow_softmax_pub, in_mask, is_priv=False)
    X_priv_sm = softmax_features(target_softmax_priv, shadow_softmax_priv, in_mask, is_priv=True)

    X_pub_full = np.column_stack([stats_pub, X_pub_sm])
    X_priv_full = np.column_stack([stats_priv, X_priv_sm])

    stats_pub = np.nan_to_num(stats_pub, nan=0.0, posinf=1000.0, neginf=0.0)
    stats_priv = np.nan_to_num(stats_priv, nan=0.0, posinf=1000.0, neginf=0.0)
    X_pub_sm = np.nan_to_num(X_pub_sm, nan=0.0, posinf=1000.0, neginf=0.0)
    X_priv_sm = np.nan_to_num(X_priv_sm, nan=0.0, posinf=1000.0, neginf=0.0)
    X_pub_full = np.nan_to_num(X_pub_full, nan=0.0, posinf=1000.0, neginf=0.0)
    X_priv_full = np.nan_to_num(X_priv_full, nan=0.0, posinf=1000.0, neginf=0.0)

    print(f"\nFeature shapes:")
    print(f"  Features Shape: pub={X_pub_full.shape}")

    def tpr_at_fpr(y_true, y_score, target_fpr=0.05):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        idx = np.searchsorted(fpr, target_fpr, side='right') - 1
        return tpr[max(0, idx)]

    def fit_predict(X_train, y_train, X_test, model_type):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        
        if model_type == "histgb":
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.02, max_depth=3, random_state=42)
        elif model_type == "logreg_l1":
            clf = LogisticRegression(max_iter=2000, C=0.5, penalty='l1', solver='liblinear', random_state=42)
        elif model_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=42)
        elif model_type == "lightgbm":
            clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
            
        clf.fit(X_tr, y_train)
        return clf.predict_proba(X_te)[:, 1]


    def cv_eval(X, y, labels, X_priv, priv_labels, model_type, n_splits=5):
        cv_scores_pub = np.zeros(len(y))
        scores_priv = np.zeros(len(priv_labels))
        
        for cls in range(NUM_CLASSES):
            idx_pub = np.where(labels == cls)[0]
            idx_priv = np.where(priv_labels == cls)[0]
            if len(idx_pub) == 0: continue

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            X_cls, y_cls = X[idx_pub], y[idx_pub]
            for tr, te in kf.split(X_cls):
                cv_scores_pub[idx_pub[te]] = fit_predict(X_cls[tr], y_cls[tr], X_cls[te], model_type)
   
            if len(idx_priv) > 0:
                scores_priv[idx_priv] = fit_predict(X_cls, y_cls, X_priv[idx_priv], model_type)
        
        return tpr_at_fpr(y, cv_scores_pub), scores_priv


    print("\n=== CV Evaluation ===")
    configs = []
    configs.append(("v5_only_HistGB",      "histgb",   stats_pub,        stats_priv))
    configs.append(("v5_only_LogReg",      "logreg",   stats_pub,        stats_priv))
    configs.append(("v6_softmax_HistGB",   "histgb",   X_pub_sm,        X_priv_sm))
    configs.append(("v6_softmax_LogReg",   "logreg",   X_pub_sm,        X_priv_sm))
    configs.append(("v6_softmax_LogReg_L1","logreg_l1",X_pub_sm,        X_priv_sm))
    configs.append(("FULL_HistGB",         "histgb",   X_pub_full,      X_priv_full))
    configs.append(("FULL_LogReg",         "logreg",   X_pub_full,      X_priv_full))
    configs.append(("FULL_LogReg_L1",      "logreg_l1",X_pub_full,      X_priv_full))
    configs.append(("FULL_MLP",            "mlp",      X_pub_full,      X_priv_full))
    configs.append(("v5_only_LightGBM",    "lightgbm", stats_pub,        stats_priv))
    configs.append(("v6_softmax_LightGBM", "lightgbm", X_pub_sm,        X_priv_sm))
    configs.append(("FULL_LightGBM",       "lightgbm", X_pub_full,      X_priv_full))

    results = {}
    for name, mtype, Xp, Xpv in configs:
        try:
            cv_tpr, priv_preds = cv_eval(Xp, pub_mem, pub_labels, Xpv, priv_labels, mtype)
            results[name] = (cv_tpr, priv_preds)
            print(f"  {name:30s}: CV TPR@5%FPR = {cv_tpr:.4f}")
        except Exception as e:
            print(f"  {name:30s}: FAILED ({e})")

    print("\nEnsemble of top-3")
    sorted_results = sorted(results.items(), key=lambda x: -x[1][0])
    top3 = sorted_results[:3]
    print(f"Top 3: {[t[0] for t in top3]}")

    ensemble_priv = np.zeros(n_priv)
    ensemble_cv = np.zeros(n_pub)

    for name, (_, p) in top3:
        ensemble_priv += pd.Series(p).rank(method='average').values / len(p)
    ensemble_priv /= len(top3 )

    print("Computing ensemble CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, te_idx in kf.split(np.arange(n_pub)):
        fold_ranks = []
        for name, _ in top3:
            cfg = [c for c in configs if c[0] == name][0]
            _, mtype, Xp, _ = cfg

            fold_preds = np.zeros(len(te_idx))
            for cls in range(NUM_CLASSES):
                te_cls_mask = (pub_labels[te_idx] == cls)
                if not te_cls_mask.any(): 
                    continue

                tr_cls_idx = np.intersect1d(np.where(pub_labels == cls)[0], tr_idx)

                te_cls_idx_global = te_idx[te_cls_mask]

                te_cls_idx_in_fold = np.where(te_cls_mask)[0]

                if len(tr_cls_idx) == 0: 
                    continue
                
                p = fit_predict(Xp[tr_cls_idx], pub_mem[tr_cls_idx], Xp[te_cls_idx_global], mtype)
                fold_preds[te_cls_idx_in_fold] = p
            
            fold_ranks.append(pd.Series(fold_preds).rank(method='average').values / len(fold_preds))

        ensemble_cv[te_idx] = np.mean(fold_ranks, axis=0)

    ens_cv_tpr = tpr_at_fpr(pub_mem, ensemble_cv)
    print(f"  Ensemble (top-3 avg) CV TPR@5%FPR: {ens_cv_tpr:.4f}")
    results["ENSEMBLE_top3"] = (ens_cv_tpr, ensemble_priv)

    best_name = max(results, key=lambda k: results[k][0])
    best_cv, best_priv = results[best_name]
    print(f"\nBEST: {best_name} (CV = {best_cv:.4f})")

    ranks = pd.Series(best_priv).rank(method='average').values
    score_01 = (ranks - 1) / (len(ranks) - 1)
    pd.DataFrame({"id": [str(int(i)) for i in priv_ds.ids], "score": score_01}).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved scores to: {OUTPUT_CSV}")

    print("\nFinal Summary")
    for name, (cv, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:30s}: CV TPR@5%FPR = {cv:.4f}")

# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)