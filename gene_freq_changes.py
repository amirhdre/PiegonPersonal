import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = "data/TCR Data with VDJ freq"
all_transitions = []

# Ensure the directory exists
if not os.path.exists(root_dir):
    print(f"Error: Directory '{root_dir}' not found.")
    exit()

for pt_folder in sorted(os.listdir(root_dir)):
    if not pt_folder.startswith("Pt"):  
        continue
    
    patient_id = pt_folder  
    patient_path = os.path.join(root_dir, pt_folder)
    
    patient_data = []  # Store data for V, D, and J genes
    
    for file in sorted(os.listdir(patient_path)):
        if not file.endswith(".csv"):
            continue
        
        parts = file.split("_")
        if len(parts) < 4:  # Ensure the expected format
            continue
        
        try:
            timepoint = int(parts[0])
            condition = parts[2]  # "F" for flare, "R" for remission
            gene_segment = parts[3].split(".")[0]  # Extract gene type (v, d, or j)
            
            if gene_segment not in ["v", "d", "j"]:  # Only process V, D, and J genes
                continue
            
            file_path = os.path.join(patient_path, file)
            df = pd.read_csv(file_path)
            df = df.set_index("gene_segment")
            
            patient_data.append((timepoint, condition, gene_segment, file, df))
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    patient_data.sort()
    for i in range(1, len(patient_data)):
        prev_timepoint, prev_condition, prev_gene, prev_file, prev_df = patient_data[i - 1]
        curr_timepoint, curr_condition, curr_gene, curr_file, curr_df = patient_data[i]
        
        transition_type = f"{prev_condition}→{curr_condition}"
        
        # Compute the difference in gene frequencies
        freq_diff = curr_df.sub(prev_df, fill_value=0)
        freq_diff = freq_diff.reset_index()
        
        # Store transition information in a single dataframe
        transition_record = {
            "Patient": patient_id,
            "Flare_Timepoint": prev_timepoint,
            "Remission_Timepoint": curr_timepoint,
            "Transition_Type": transition_type,
            "Gene_Type": prev_gene,
            "Flare_File": prev_file,
            "Remission_File": curr_file,
        }
        
        for gene, change in freq_diff.set_index("gene_segment").iterrows():
            transition_record[gene] = np.log1p(change["freq"])  # Log normalize
        
        all_transitions.append(transition_record)

# Convert all transitions into a single DataFrame
transitions_df = pd.DataFrame(all_transitions)

# Save to CSV
transitions_df.to_csv("vdj_gene_transitions.csv", index=False)
print("Saved V, D, and J gene transition data to 'vdj_gene_transitions.csv'.")

# Perform PCA on log-normalized frequency changes
feature_columns = [col for col in transitions_df.columns if col not in ["Patient", "Flare_Timepoint", "Remission_Timepoint", "Transition_Type", "Gene_Type", "Flare_File", "Remission_File"]]
transitions_df.fillna(0).to_csv("raw_freq_vdj_data", index=False)
X = transitions_df[feature_columns].fillna(0)
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

# Create a DataFrame for the filtered data
filtered_df = transitions_df.iloc[inlier_indices].copy()
filtered_df["PCA1"] = principal_components[:, 0]
filtered_df["PCA2"] = principal_components[:, 1]

# Save the updated DataFrame with PCA results
filtered_df.to_csv("vdj_gene_transitions_with_pca_filtered_isoforest.csv", index=False)
print("Saved PCA results (filtered by Isolation Forest) to 'vdj_gene_transitions_with_pca_filtered_isoforest.csv'.")

# Identify the genes that contribute most to each principal component
gene_contributions = pd.DataFrame(
    pca.components_.T,
    index=feature_columns,
    columns=["PC1", "PC2"]
)

top_genes_pc1 = gene_contributions["PC1"].abs().sort_values(ascending=False).head(10)
top_genes_pc2 = gene_contributions["PC2"].abs().sort_values(ascending=False).head(10)

print("\nTop 10 Genes Contributing to PC1:")
print(top_genes_pc1)

print("\nTop 10 Genes Contributing to PC2:")
print(top_genes_pc2)

# Define custom colors for transitions
palette = {"F→F": "blue", "R→R": "blue", "F→R": "red", "R→F": "red"}

# Plot PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_df, x="PCA1", y="PCA2", hue="Transition_Type", palette=palette, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Log-Normalized V, D, J Gene Frequency Changes Colored by Transition Type (Filtered by Isolation Forest)")
plt.legend(title="Transition Type")
plt.grid(True)
plt.show()

# Train a Random Forest classifier
X_train, X_test, y_train, y_test = train_test_split(X_filtered, filtered_df["Transition_Type"], test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{class_report}")
# Extract top 10 genes from PC1
top_genes = top_genes_pc1.index.tolist()

# Filter the dataset to include only the top genes
gene_freq_data = transitions_df.melt(id_vars=["Transition_Type"], value_vars=top_genes, var_name="Gene", value_name="Log-Normalized Frequency")
print(gene_freq_data.head())
# Map transition types to flare and remission
gene_freq_data["Condition"] = gene_freq_data["Transition_Type"].apply(lambda x: "Flare" if "F" in x.split("→")[0] else "Remission")

# Plot gene frequency distribution
plt.figure(figsize=(12, 6))
sns.barplot(data=gene_freq_data, x="Gene", y="Log-Normalized Frequency", hue="Condition", palette={"Flare": "red", "Remission": "blue"})
plt.xticks(rotation=45)
plt.xlabel("Gene")
plt.ylabel("Log-Normalized Frequency")
plt.title("Top 10 Genes Contributing to PC1: Frequency in Flare vs. Remission")
plt.legend(title="Condition")
plt.show()
