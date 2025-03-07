import pandas as pd

# Load the dataframe and ignore the first unnamed column
df = pd.read_csv("gene_frequency_changes_by_transition.csv", index_col=0)

# Identify metadata columns (including "transition" which should be excluded from sorting)
metadata_cols = ['patient', 'sample', 'state_t1', 'state_t2', 'transition']  # Modify if needed
gene_cols = [col for col in df.columns if col not in metadata_cols]  # Gene columns

# Initialize a list to store results
top_genes_per_row = []

# Iterate through each row (each row is a transition type)
for idx, row in df.iterrows():
    # Compute absolute changes for all genes in this row (excluding 'transition')
    abs_changes = row[gene_cols].abs()
    
    # Get the top 5 genes with the highest absolute change
    top_genes = abs_changes.nlargest(5)

    # Store results in a structured format
    result = {
        'index': idx,  # Store the row index
        'transition': row['transition'],  # Keep the transition column
        'top_gene_1': top_genes.index[0],
        'top_gene_1_change': top_genes.values[0],
        'top_gene_2': top_genes.index[1],
        'top_gene_2_change': top_genes.values[1],
        'top_gene_3': top_genes.index[2],
        'top_gene_3_change': top_genes.values[2],
        'top_gene_4': top_genes.index[3],
        'top_gene_4_change': top_genes.values[3],
        'top_gene_5': top_genes.index[4],
        'top_gene_5_change': top_genes.values[4]
    }

    top_genes_per_row.append(result)

# Convert results into a DataFrame
top_genes_df = pd.DataFrame(top_genes_per_row)

top_genes_df.to_csv("top_genes.csv")
# Display the results
print(top_genes_df)
