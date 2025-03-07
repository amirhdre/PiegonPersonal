import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_pca():
    # Load the data
    df = pd.read_csv('tcr_frequency_changes.csv')

    # Extract the features (all columns except for 'patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2')
    feature_columns = [col for col in df.columns if col not in ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']]
    data = df[feature_columns]

    # Fill NaN values with 0
    data = data.fillna(0)

    # Log normalize the data
    data_log = np.log1p(data)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log)

    # Apply PCA
    pca = PCA(n_components=2)  # You can change the number of components if needed
    pca_result = pca.fit_transform(data_scaled)

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    # Add patient and file info back for visualization
    pca_df['patient'] = df.loc[data_log.index, 'patient']
    pca_df['file_t1'] = df.loc[data_log.index, 'file_t1']
    pca_df['file_t2'] = df.loc[data_log.index, 'file_t2']
    
    # Add the transition type (state transition between sample_t1 and sample_t2)
    transition = []
    for i, row in df.iterrows():
        state1 = row['state_t1']
        state2 = row['state_t2']
        transition.append(f"{state1} to {state2}")
    pca_df['transition'] = transition

    # Calculate the z-scores of the PCA components to identify outliers
    pca_df['zscore_pc1'] = (pca_df['PC1'] - pca_df['PC1'].mean()) / pca_df['PC1'].std()
    pca_df['zscore_pc2'] = (pca_df['PC2'] - pca_df['PC2'].mean()) / pca_df['PC2'].std()

    # Filter out points with a z-score above a threshold (e.g., 3 or -3)
    pca_df_cleaned = pca_df[(np.abs(pca_df['zscore_pc1']) < 3) & (np.abs(pca_df['zscore_pc2']) < 3)]

    # Plot the results after removing outliers, color by transition type
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_df_cleaned['PC1'], pca_df_cleaned['PC2'], c=pca_df_cleaned['transition'].astype('category').cat.codes, cmap='viridis')
    plt.colorbar(scatter, label='Transition Type')
    plt.title('PCA of TCR Frequency Changes (Outliers Removed)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Optionally save the PCA result to a CSV
    pca_df_cleaned.to_csv('pca_results_cleaned_with_transitions.csv', index=False)
    print("PCA results (outliers removed) saved to pca_results_cleaned_with_transitions.csv")

if __name__ == "__main__":
    run_pca()
