import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, auc,
                            precision_recall_curve)
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
import lightgbm as lgb
import shap
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)

 
# Function to load and preprocess TCR data
def load_and_preprocess_tcr_data(file_path, target_column='Transition_Type'):
    """
    Load and preprocess TCR data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing TCR data
    target_column : str, default='Transition_Type'
        Name of the target column for classification
        
    Returns:
    --------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target variable
    feature_names : list
        List of feature names
    metadata_columns : list
        List of metadata columns (non-features)
    """
    try:
        # Load the data
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Print basic information about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"First few rows of the dataset:")
        print(df.head())
        
        # Identify metadata columns (non-features)
        metadata_columns = ["Patient", "Flare_Timepoint", "Remission_Timepoint", 
                            "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"]
        
        # Extract feature columns
        feature_columns = [col for col in df.columns if col not in metadata_columns]
        
        # Replace any NaN values with 0
        X = df[feature_columns].fillna(0)
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        # For this classification, we want to predict transition types
        # Binary classification: is it a transition (F->R, R->F) or not (F->F, R->R)
        print(f"Unique values in {target_column}: {df[target_column].unique()}")
        
        # Create binary target: 1 for transitions (F->R, R->F), 0 for non-transitions (F->F, R->R)
        y = df[target_column].astype(str).str.strip().isin(["F->R", "R->F"]).astype(int)
        y.name = "is_transition"
        
        print(f"Target distribution:")
        print(y.value_counts())
        
        return X, y, feature_columns, metadata_columns, df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Function to handle outliers
def handle_outliers(X, contamination=0.05):
    """
    Detect and handle outliers using Isolation Forest.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    contamination : float, default=0.05
        The proportion of outliers in the data set
        
    Returns:
    --------
    X_filtered : numpy.ndarray
        Filtered features without outliers
    inlier_indices : numpy.ndarray
        Indices of inliers
    """
    # Standardize the data
    if isinstance(X, pd.DataFrame):
        X_scaled = StandardScaler().fit_transform(X)
    else:
        X_scaled = StandardScaler().fit_transform(X)
    
    # Apply Isolation Forest to detect outliers
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_predictions = iso_forest.fit_predict(X_scaled)
    
    # Filter out the outliers
    inlier_indices = np.where(outlier_predictions == 1)[0]
    X_filtered = X_scaled[inlier_indices]
    
    print(f"Removed {len(X_scaled) - len(X_filtered)} outliers "
          f"({(len(X_scaled) - len(X_filtered)) / len(X_scaled) * 100:.2f}% of data).")
    
    return X_filtered, inlier_indices

# Function to train a LightGBM classifier
def train_lightgbm_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train a LightGBM classifier on TCR data.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    model : LGBMClassifier
        Trained LightGBM model
    X_train, X_test, y_train, y_test : arrays
        Train and test splits of the data
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Initialize the LightGBM classifier
    model = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=random_state
    )
    
    # Create a validation set for early stopping
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    # Train the model with early stopping
    model.fit(
        X_train_inner, y_train_inner,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        early_stopping_rounds=50,
        verbose=100
    )
    
    # Retrain on the entire training set with the best iteration
    best_iterations = model.best_iteration_
    model = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=best_iterations,
        verbose=-1,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the LightGBM model on the test set.
    
    Parameters:
    -----------
    model : LGBMClassifier
        Trained LightGBM model
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        Test target variable
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    y_pred : numpy.ndarray
        Predicted labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_pred_proba

# Function to perform cross-validation
def perform_cross_validation(X, y, n_splits=5, random_state=42):
    """
    Perform cross-validation on the LightGBM model.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    cv_results : dict
        Dictionary of cross-validation results
    """
    # Initialize the LightGBM classifier
    model = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=random_state
    )
    
    # Initialize StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Metrics to track
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = defaultdict(list)
    
    # Perform cross-validation
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model
        model.fit(X_train_cv, y_train_cv)
        
        # Make predictions
        y_pred_cv = model.predict(X_test_cv)
        
        # Calculate metrics
        cv_results['accuracy'].append(accuracy_score(y_test_cv, y_pred_cv))
        cv_results['precision'].append(precision_score(y_test_cv, y_pred_cv))
        cv_results['recall'].append(recall_score(y_test_cv, y_pred_cv))
        cv_results['f1'].append(f1_score(y_test_cv, y_pred_cv))
        
        print(f"Fold {fold+1}/{n_splits} - "
              f"Accuracy: {cv_results['accuracy'][-1]:.4f}, "
              f"Precision: {cv_results['precision'][-1]:.4f}, "
              f"Recall: {cv_results['recall'][-1]:.4f}, "
              f"F1: {cv_results['f1'][-1]:.4f}")
    
    # Calculate mean and std for each metric
    cv_summary = {}
    for metric in metrics:
        cv_summary[f'{metric}_mean'] = np.mean(cv_results[metric])
        cv_summary[f'{metric}_std'] = np.std(cv_results[metric])
    
    print("\nCross-Validation Summary:")
    for metric in metrics:
        print(f"{metric.capitalize()}: {cv_summary[f'{metric}_mean']:.4f} ± {cv_summary[f'{metric}_std']:.4f}")
    
    return cv_results, cv_summary

# Function to plot feature importance
def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance of the LightGBM model.
    
    Parameters:
    -----------
    model : LGBMClassifier
        Trained LightGBM model
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to plot
    """
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top N features
    top_features = importance_df.head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('tcr_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, target_names=['Non-Transition', 'Transition']):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_test : pandas.Series or numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    target_names : list, default=['Non-Transition', 'Transition']
        Names of target classes
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('tcr_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_test : pandas.Series or numpy.ndarray
        True labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('tcr_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

# Function to visualize SHAP values
def plot_shap_values(model, X, feature_names, plot_type='bar', max_display=20):
    """
    Plot SHAP values for feature importance explanation.
    
    Parameters:
    -----------
    model : LGBMClassifier
        Trained LightGBM model
    X : pandas.DataFrame or numpy.ndarray
        Features
    feature_names : list
        List of feature names
    plot_type : str, default='bar'
        Type of SHAP plot ('bar', 'dot', 'violin', etc.)
    max_display : int, default=20
        Maximum number of features to display
    """
    try:
        # Create a DataFrame for SHAP analysis if X is a numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        
        # For binary classification, shap_values is a list with one element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Plot SHAP values
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_df, plot_type='bar', max_display=max_display, show=False)
            plt.title('SHAP Feature Importance')
        else:
            shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
            plt.title('SHAP Summary Plot')
        
        plt.tight_layout()
        plt.savefig(f'tcr_shap_{plot_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    except Exception as e:
        print(f"Error plotting SHAP values: {e}")

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_test, y_pred_proba):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_test : pandas.Series or numpy.ndarray
        True labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('tcr_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pr_auc

# Function to analyze feature distributions between classes
def plot_feature_distributions(X, y, feature_names, top_features, n_features=5):
    """
    Plot distributions of top features between classes.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    feature_names : list
        List of feature names
    top_features : pandas.DataFrame
        DataFrame of feature importance
    n_features : int, default=5
        Number of top features to plot
    """
    # Create a DataFrame for analysis
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
    
    # Add target variable
    X_df['target'] = y
    
    # Get top N features
    top_n_features = top_features.head(n_features)['Feature'].tolist()
    
    # Plot distributions
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    
    for i, feature in enumerate(top_n_features):
        sns.histplot(
            data=X_df, x=feature, hue='target', 
            element='step', stat='density', common_norm=False,
            ax=axes[i]
        )
        axes[i].set_title(f'Distribution of {feature} by Class')
        axes[i].legend(['Non-Transition', 'Transition'])
    
    plt.tight_layout()
    plt.savefig('tcr_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main function
def main(file_path="raw_freq_vdj_data.csv"):
    """
    Main function to run the TCR classification pipeline.
    
    Parameters:
    -----------
    file_path : str, default="raw_freq_vdj_data.csv"
        Path to the CSV file containing TCR data
    """
    print("=== LightGBM Classifier for TCR Transition Data ===")
    
    # Load and preprocess data
    X, y, feature_names, metadata_columns, df_original = load_and_preprocess_tcr_data(file_path)
    
    # Handle outliers
    X_filtered, inlier_indices = handle_outliers(X)
    y_filtered = y.iloc[inlier_indices]
    
    # Train the model
    print("\n=== Training LightGBM Classifier ===")
    model, X_train, X_test, y_train, y_test = train_lightgbm_classifier(X_filtered, y_filtered)
    
    # Evaluate the model
    print("\n=== Evaluating Model Performance ===")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Perform cross-validation
    print("\n=== Performing Cross-Validation ===")
    cv_results, cv_summary = perform_cross_validation(X_filtered, y_filtered)
    
    # Plot results
    print("\n=== Generating Visualizations ===")
    
    # Plot feature importance
    print("Plotting feature importance...")
    importance_df = plot_feature_importance(model, feature_names, top_n=20)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    roc_auc = plot_roc_curve(y_test, y_pred_proba)
    
    # Plot precision-recall curve
    print("Plotting precision-recall curve...")
    pr_auc = plot_precision_recall_curve(y_test, y_pred_proba)
    
    # Plot SHAP values
    print("Plotting SHAP values...")
    # Convert X_test to DataFrame with feature names for SHAP analysis
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    plot_shap_values(model, X_test_df, feature_names, plot_type='bar')
    plot_shap_values(model, X_test_df, feature_names, plot_type='dot')
    
    # Plot feature distributions
    print("Plotting feature distributions...")
    X_test_orig = pd.DataFrame(X_test, columns=feature_names)
    plot_feature_distributions(X_test_orig, y_test, feature_names, importance_df)
    
    # Save feature importance
    importance_df.to_csv('tcr_feature_importance.csv', index=False)
    
    print("\n=== Analysis Complete ===")
    print("All plots and results have been saved to the current directory.")
    
    return model, metrics, cv_summary, importance_df

# Run the main function
if __name__ == "__main__":
    main("raw_freq_vdj_data.csv")
