import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Silence noisy sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Base project directory (PostureAI folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "dataset", "workout_pose_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "workout_pose_model.joblib")


def main():
    print(f"📂 Loading pose dataset: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print("❌ Dataset file not found. Please generate workout_pose_dataset.csv first.")
        return

    df = pd.read_csv(DATA_PATH)

    # ---- FEATURE SELECTION ----
    # Keep ONLY numeric columns (angles etc.), drop non-numeric like image paths.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "label" in numeric_cols:
        numeric_cols.remove("label")

    if not numeric_cols:
        print("❌ No numeric feature columns found. Check workout_pose_dataset.csv.")
        return

    feature_cols = numeric_cols
    print(f"➡️ Using feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values
    y = df["label"].values

    print(f"✅ Dataset loaded: {len(df)} samples")
    print("➡️ Classes:", sorted(np.unique(y)))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # Train model
    print("🚀 Training RandomForest model...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    # Save both model + metadata
    pack = {
        "model": clf,
        "class_names": sorted(np.unique(y)),
        "feature_names": feature_cols,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pack, MODEL_PATH)

    print(f"\n✅ Model saved successfully to:\n   {MODEL_PATH}")


if __name__ == "__main__":
    main()
