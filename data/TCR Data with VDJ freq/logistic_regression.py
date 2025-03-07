import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
df = pd.read_csv("tcr_frequency_changes_w_transition.csv")

# Identify columns
metadata_cols = ['patient', 'sample', 'state_t1', 'state_t2', 'transition', 'file_t1', 'file_t2']
gene_cols = [col for col in df.columns if col not in metadata_cols]

# Extract features (gene frequency changes) and target (transition labels)
X = df[gene_cols]  # Features
y = df['transition']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Impute missing values in the features (only numeric columns)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Apply SMOTE to handle class imbalance in the training set
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Compute class weights to handle imbalanced classes in Logistic Regression
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
class_weight_dict = dict(zip(np.unique(y_train_res), class_weights))

# Set up Logistic Regression with Grid Search for hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}  # Regularization parameter grid
grid_search = GridSearchCV(LogisticRegression(class_weight=class_weight_dict, max_iter=10000), 
                           param_grid, cv=3, n_jobs=-1, scoring='accuracy')

# Fit Grid Search to the resampled training data
grid_search.fit(X_train_res, y_train_res)

# Get the best model from the Grid Search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Display the results
print("Best Parameters from Grid Search:", grid_search.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
