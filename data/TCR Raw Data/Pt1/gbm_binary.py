import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess

def ensure_libomp():
    try:
        subprocess.run(['brew', 'install', 'libomp'], check=True)
    except Exception as e:
        print("Failed to install libomp automatically. Please install it manually using: brew install libomp")

def train_gbm(file_path):
    df = pd.read_csv(file_path)
    
    # Define feature columns (excluding identifiers and labels)
    feature_columns = [col for col in df.columns if col not in ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']]
    data = df[feature_columns].fillna(0)
    
    # Define binary transition target variable
    df['transition'] = df.apply(lambda row: "Same" if row['state_t1'] == row['state_t2'] else "Change", axis=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['transition'])  # Encode transition types numerically
    
    # Standardize feature data
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Ensure OpenMP library is available for XGBoost
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    ensure_libomp()
    
    # Train XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=np.arange(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(label_encoder.classes_)), labels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.show()
    
if __name__ == "__main__":
    file_path = 'tcr_frequency_changes.csv'  # Replace with actual file path
    train_gbm(file_path)