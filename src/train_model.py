"""
Training script for the churn models.
I tried a few different algorithms and also did a quick Optuna run for XGBoost.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from data_preprocessing import load_raw_data, create_features, encode_and_split


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print the usual metrics I care about for churn."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_prob


def train_all_models(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Train several models and keep the best one based on PR-AUC
    (better than ROC when classes are imbalanced).
    """
    if use_smote:
        print("Applying SMOTE on the training set...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("New train shape:", X_train.shape)
    
    models = {}
    
    # 1. Logistic Regression (baseline)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    models["logistic"] = lr
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    
    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # 3. XGBoost
    print("\nTraining XGBoost...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    models["xgboost"] = xgb_clf
    evaluate_model(xgb_clf, X_test, y_test, "XGBoost")
    
    # 4. LightGBM
    print("\nTraining LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train)
    models["lightgbm"] = lgb_clf
    evaluate_model(lgb_clf, X_test, y_test, "LightGBM")
    
    return models


def save_best_model(models, X_test, y_test, output_dir="models"):
    """Pick the model with highest PR-AUC and save it."""
    best_name = None
    best_score = -1
    best_model = None
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        score = average_precision_score(y_test, y_prob)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"best_model_{best_name}.joblib")
    joblib.dump(best_model, path)
    print(f"\nBest model: {best_name} (PR-AUC = {best_score:.4f})")
    print(f"Saved to {path}")
    return best_model, best_name


if __name__ == "__main__":
    print("Loading and preparing data...")
    raw = load_raw_data()
    df = create_features(raw)
    X_train, X_test, y_train, y_test, encoders = encode_and_split(df)
    
    # also save the feature names for later
    joblib.dump(list(X_train.columns), "models/feature_names.joblib")
    joblib.dump(encoders, "models/encoders.joblib")
    
    models = train_all_models(X_train, y_train, X_test, y_test, use_smote=True)
    best_model, best_name = save_best_model(models, X_test, y_test)
    
    print("\nDone.")
