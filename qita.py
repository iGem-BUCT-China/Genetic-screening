import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from rdkit.Avalon import pyAvalonTools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
import os

# Read Excel file into DataFrame
data_path = r"C:/Users/86158/Desktop/data.xlsx"
data = pd.read_excel(data_path)

# Helper function to compute fingerprints
def compute_fingerprint(smlies, fingerprint_type):
    mol = Chem.MolFromSmiles(smlies)
    if not mol:
        return None
    if fingerprint_type == 'MACCS':
        return list(MACCSkeys.GenMACCSKeys(mol))
    elif fingerprint_type == 'ECFP4':
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=False).ToBitString())
    elif fingerprint_type == 'FCFP4':
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=True).ToBitString())
    elif fingerprint_type == 'RDK':
        return list(RDKFingerprint(mol, maxPath=5, fpSize=1024).ToBitString())
    elif fingerprint_type == 'Avalon':
        return list(pyAvalonTools.GetAvalonFP(mol, nBits=1024).ToBitString())

# Compute fingerprints and add them to DataFrame
fingerprints = ['MACCS', 'ECFP4', 'FCFP4', 'RDK', 'Avalon']
for fingerprint in fingerprints:
    data[fingerprint] = data['smlies'].apply(lambda smlies: compute_fingerprint(smlies, fingerprint))

# Drop rows with missing fingerprints
data = data.dropna(subset=fingerprints)

# Define activity thresholds
thresholds = [100, 200, 500, 1000, 1500]

# Function to process fingerprints and perform XGBoost classification
def process_fingerprint(fingerprint_name, fingerprint_col):
    results = []
    for threshold in thresholds:
        data['Active'] = data['Standard Value'] <= threshold
        X = pd.DataFrame(data[fingerprint_col].tolist())
        y = data['Active']

        for seed in [701, 810, 520]:
            np.random.seed(seed)

            for ratio in [0.75, 0.80]:
                train_size = int(len(X) * ratio)
                indices = np.random.permutation(len(X))
                train_idx, test_idx = indices[:train_size], indices[train_size:]
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Feature selection
                cor_matrix = np.corrcoef(X_train_scaled.T, y_train)[-1][:-1]
                pearson_selected = np.abs(cor_matrix) > 0.1
                X_train_pearson = X_train_scaled[:, pearson_selected]
                X_test_pearson = X_test_scaled[:, pearson_selected]

                # Recursive feature elimination with cross-validation
                xgb = XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='logloss')
                selector = RFECV(xgb, step=1, cv=3, n_jobs=-1)
                selector.fit(X_train_pearson, y_train)
                X_train_rfe = selector.transform(X_train_pearson)
                X_test_rfe = selector.transform(X_test_pearson)

                # Train XGBoost model
                xgb.fit(X_train_rfe, y_train)
                y_train_pred = xgb.predict(X_train_rfe)
                y_test_pred = xgb.predict(X_test_rfe)

                # Calculate metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                mcc = matthews_corrcoef(y_test, y_test_pred)

                try:
                    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
                except ValueError:
                    tn_train, fp_train, fn_train, tp_train = 0, 0, 0, 0

                try:
                    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
                except ValueError:
                    tn, fp, fn, tp = 0, 0, 0, 0

                se_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0
                sp_train = tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else 0

                se = tp / (tp + fn) if (tp + fn) > 0 else 0
                sp = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                total_cv_tp = total_cv_fp = total_cv_tn = total_cv_fn = 0
                cv_scores = []

                for train_idx_cv, test_idx_cv in cv.split(X_train_rfe, y_train):
                    X_train_cv, X_test_cv = X_train_rfe[train_idx_cv], X_train_rfe[test_idx_cv]
                    y_train_cv, y_test_cv = y_train.iloc[train_idx_cv], y_train.iloc[test_idx_cv]
                    xgb.fit(X_train_cv, y_train_cv)
                    y_test_cv_pred = xgb.predict(X_test_cv)
                    cv_scores.append(accuracy_score(y_test_cv, y_test_cv_pred))

                    tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_test_cv, y_test_cv_pred).ravel()
                    total_cv_tp += tp_cv
                    total_cv_fp += fp_cv
                    total_cv_tn += tn_cv
                    total_cv_fn += fn_cv

                # Average cross-validation metrics
                cv_accuracy = np.mean(cv_scores)
                cv_se_avg = total_cv_tp / (total_cv_tp + total_cv_fn) if (total_cv_tp + total_cv_fn) > 0 else 0
                cv_sp_avg = total_cv_tn / (total_cv_tn + total_cv_fp) if (total_cv_tn + total_cv_fp) > 0 else 0
                cv_b_acc = (cv_se_avg + cv_sp_avg) / 2
                cv_mcc = matthews_corrcoef(
                    np.concatenate([y_train.iloc[test_idx_cv] for train_idx_cv, test_idx_cv in cv.split(X_train_rfe, y_train)]),
                    np.concatenate([xgb.predict(X_train_rfe[test_idx_cv]) for train_idx_cv, test_idx_cv in cv.split(X_train_rfe, y_train)])
                )

                # Store results
                results.append({
                    'n_features': X_train_rfe.shape[1],
                    'algorithm': 'XGBoost',
                    'params': "None",
                    'tr_accuracy': f"{train_acc * 100:.2f}%",
                    'tr_b_acc': f"{(se_train + sp_train) / 2 * 100:.2f}%",
                    'tr_mcc': f"{matthews_corrcoef(y_train, y_train_pred):.4f}",
                    'tr_tp': tp_train,
                    'tr_fp': fp_train,
                    'tr_tn': tn_train,
                    'tr_fn': fn_train,
                    'tr_se': f"{se_train * 100:.2f}%",
                    'tr_sp': f"{sp_train * 100:.2f}%",
                    'cv_accuracy': f"{cv_accuracy * 100:.2f}%",
                    'cv_b_acc': f"{cv_b_acc * 100:.2f}%",
                    'cv_mcc': f"{cv_mcc:.4f}",
                    'cv_tp': total_cv_tp,
                    'cv_fp': total_cv_fp,
                    'cv_tn': total_cv_tn,
                    'cv_fn': total_cv_fn,
                    'cv_se': f"{cv_se_avg * 100:.2f}%",
                    'cv_sp': f"{cv_sp_avg * 100:.2f}%",
                    'te_accuracy': f"{test_acc * 100:.2f}%",
                    'te_b_acc': f"{(se + sp) / 2 * 100:.2f}%",
                    'te_mcc': f"{mcc:.4f}",
                    'te_tp': tp,
                    'te_fp': fp,
                    'te_tn': tn,
                    'te_fn': fn,
                    'te_se': f"{se * 100:.2f}%",
                    'te_sp': f"{sp * 100:.2f}%",
                    'notes': f"Threshold: {threshold}, Seed: {seed}, Ratio: {ratio}"
                })

    return results

# Compute and save results for each fingerprint type
output_dir = r"C:/Users/86158/Desktop/2"
os.makedirs(output_dir, exist_ok=True)

for name, col in zip(fingerprints, fingerprints):
    result = process_fingerprint(name, col)
    results_df = pd.DataFrame(result)
    results_df.to_excel(os.path.join(output_dir, f"xgboost_results_{name}.xlsx"), index=False)
