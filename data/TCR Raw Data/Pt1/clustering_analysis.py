import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

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
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['patient'].astype('category').cat.codes, cmap='tab10')
    plt.colorbar(scatter, label='Patient')
    plt.title('PCA of TCR Frequency Changes (Colored by Patient)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def run_tsne(file_path):
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col not in ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']]
    data = df[feature_columns].fillna(0)
    data_log = np.log1p(data)
    data_scaled = StandardScaler().fit_transform(data_log)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)
    
    tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
    tsne_df['patient'] = df['patient']
    tsne_df['transition_category'] = df.apply(lambda row: 'Stable' if row['state_t1'] == row['state_t2'] else 'State Change', axis=1)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['transition_category'].astype('category').cat.codes, cmap='coolwarm')
    plt.colorbar(scatter, label='Transition Category')
    plt.title('t-SNE of TCR Frequency Changes (Colored by Transition)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['patient'].astype('category').cat.codes, cmap='tab10')
    plt.colorbar(scatter, label='Patient')
    plt.title('t-SNE of TCR Frequency Changes (Colored by Patient)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def run_umap(file_path):
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col not in ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']]
    data = df[feature_columns].fillna(0)
    data_log = np.log1p(data)
    data_scaled = StandardScaler().fit_transform(data_log)
    
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_result = reducer.fit_transform(data_scaled)
    
    umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
    umap_df['patient'] = df['patient']
    umap_df['transition_category'] = df.apply(lambda row: 'Stable' if row['state_t1'] == row['state_t2'] else 'State Change', axis=1)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=umap_df['transition_category'].astype('category').cat.codes, cmap='plasma')
    plt.colorbar(scatter, label='Transition Category')
    plt.title('UMAP of TCR Frequency Changes (Colored by Transition)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=umap_df['patient'].astype('category').cat.codes, cmap='tab10')
    plt.colorbar(scatter, label='Patient')
    plt.title('UMAP of TCR Frequency Changes (Colored by Patient)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

if __name__ == "__main__":
    file_path = 'tcr_frequency_changes.csv'  # Replace with the actual file path
    run_pca(file_path)
    run_tsne(file_path)
    run_umap(file_path)