import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import numpy as np
df=pd.read_csv("/Users/matthewfarah/pigeon/raw_freq_vdj_data.csv")
print(df.head())
df.to_csv("raw_freq_vdj_data.csv", index=False)

feature_columns = [col for col in df.columns if col not in ["Patient", "Flare_Timepoint", "Remission_Timepoint", "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"]]
X = df[feature_columns].fillna(0)
X = StandardScaler().fit_transform(X)

# Apply Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=0.05)
outlier_predictions = iso_forest.fit_predict(X)

# Filter out the outliers
inlier_indices = np.where(outlier_predictions == 1)[0]
X_filtered = X[inlier_indices]

# Perform PCA on the filtered data
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_filtered)

loadings = pca.components_

# Convert to DataFrame using actual feature names
# Convert to DataFrame using actual feature names
loadings_df = pd.DataFrame(loadings, columns=df[feature_columns].columns, 
                           index=[f'PC{i+1}' for i in range(loadings.shape[0])])

# Sort features by PC1 loadings
sorted_columns = loadings_df.loc['PC1', :].sort_values(ascending=False).index
loadings_df = loadings_df[sorted_columns]

# Extract top 10 and bottom 10 features based on PC1 and append additional columns
pc1_top10_features = ["Patient", "Flare_Timepoint", "Remission_Timepoint", "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"]+ sorted_columns[:10].tolist()
pc1_bottom10_features = ["Patient", "Flare_Timepoint", "Remission_Timepoint", "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"]+ sorted_columns[-10:].tolist()

# Extract corresponding DataFrames
df_pc1_top10_features = df[pc1_top10_features]
df_pc1_bottom10_features = df[pc1_bottom10_features]

print(df_pc1_top10_features.head())
print(df_pc1_bottom10_features.head())

import matplotlib.pyplot as plt
import numpy as np

# Convert to DataFrame using actual feature names
loadings_df = pd.DataFrame(loadings, columns=df[feature_columns].columns, 
                           index=[f'PC{i+1}' for i in range(loadings.shape[0])])

# Sort features by PC1 loadings
sorted_columns = loadings_df.loc['PC1', :].sort_values(ascending=False).index
loadings_df = loadings_df[sorted_columns]

# Extract top 10 and bottom 10 features based on PC1 and append additional columns
pc1_top10_features = ["Patient", "Flare_Timepoint", "Remission_Timepoint", "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"]+ sorted_columns[:10].tolist() 
pc1_bottom10_features = ["Patient", "Flare_Timepoint", "Remission_Timepoint", "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"] +sorted_columns[-10:].tolist() 
# Extract corresponding DataFrames
df_ff_rr = df[df["Transition_Type"].astype(str).str.strip().isin(["F->F", "R->R"])]
df_fr_rf = df[df["Transition_Type"].astype(str).str.strip().isin(["F->R", "R->F"])]
df_ff_rr_top10=df_ff_rr[pc1_top10_features]

# Compute mean and standard deviation for each group
def compute_stats(df_subset, features):
    means = df_subset[features].mean()
    stds = df_subset[features].std()
    return means, stds

means_ff_rr, stds_ff_rr = compute_stats(df_ff_rr, sorted_columns[:10])
means_fr_rf, stds_fr_rf = compute_stats(df_fr_rf, sorted_columns[:10])

# Plot mean and standard deviation
fig, ax = plt.subplots(figsize=(12, 6))
x_labels = sorted_columns[:10]
x = np.arange(len(x_labels))
width = 0.35

ax.bar(x - width/2, means_ff_rr, yerr=stds_ff_rr, capsize=5, label='F->F, R->R', alpha=0.7)
ax.bar(x + width/2, means_fr_rf, yerr=stds_fr_rf, capsize=5, label='F->R, R->F', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("Mean PC1 Feature Value")
ax.set_title("Top 10 Features by PC1 Loading Segmented by Transition Type")
ax.legend()

plt.tight_layout()
plt.show()

# Save to CSV
loadings_df.to_csv('pca_loadings.csv', index=True)

print("Principal Component Coefficients (Loadings):")
print(loadings_df)