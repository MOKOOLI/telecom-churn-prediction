"""
Simple prediction helper.
You can load the saved model and score new customers.
"""

import joblib
import pandas as pd
from data_preprocessing import create_features


def load_artifacts(model_path="models/best_model_xgboost.joblib"):
    model = joblib.load(model_path)
    feature_names = joblib.load("models/feature_names.joblib")
    encoders = joblib.load("models/encoders.joblib")
    return model, feature_names, encoders


def prepare_single_customer(customer_dict, encoders, feature_names):
    """
    Take a dict of raw customer attributes and turn it into the
    same feature vector the model expects.
    """
    df = pd.DataFrame([customer_dict])
    df = create_features(df)
    
    # encode categoricals the same way we did at training time
    for col, le in encoders.items():
        if col in df.columns:
            # handle unseen labels gracefully
            df[col] = df[col].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    
    # make sure column order matches
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


if __name__ == "__main__":
    # example usage
    model, feature_names, encoders = load_artifacts()
    print("Model loaded. Features:", len(feature_names))
