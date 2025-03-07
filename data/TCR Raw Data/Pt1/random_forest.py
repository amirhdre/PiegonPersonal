import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_pca(file_path):
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col not in ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']]
    data = df[feature_columns].fillna(0)
    data_log = np.log1p(data)
    data_scaled = StandardScaler().fit_transform(data_log)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['patient'] = df['patient']
    pca_df['transition'] = df.apply(lambda row: f"{row['state_t1']} to {row['state_t2']}", axis=1)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['transition'].astype('category').cat.codes, cmap='viridis')
    plt.colorbar(scatter, label='Transition Type')
    plt.title('PCA of TCR Frequency Changes (Colored by Transition)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def train_random_forest(file_path):
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col not in ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']]
    data = df[feature_columns].fillna(0)
    data_log = np.log1p(data)
    data_scaled = StandardScaler().fit_transform(data_log)
    
    df['transition'] = df.apply(lambda row: f"{row['state_t1']} to {row['state_t2']}", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, df['transition'], test_size=0.4, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    file_path = 'tcr_frequency_changes.csv'  # Replace with the actual file path
    run_pca(file_path)
    train_random_forest(file_path)
