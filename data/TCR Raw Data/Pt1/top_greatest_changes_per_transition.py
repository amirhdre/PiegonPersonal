import pandas as pd

# Load your dataframe (assuming it's already loaded as `df`)
df=pd.read_csv("tcr_frequency_changes.csv")

# Melt the dataframe to convert gene columns into rows
df_long = df.melt(id_vars=['state_t1', 'state_t2'], var_name='gene', value_name='change_in_frequency')

# Initialize a dictionary to store the top 5 genes for each transition
top_genes_by_transition = {}

# Get unique transitions
transitions = df_long[['state_t1', 'state_t2']].drop_duplicates()

# Loop through each unique transition type
for _, row in transitions.iterrows():
    state_t1, state_t2 = row['state_t1'], row['state_t2']
    
    # Filter for the specific transition
    transition_df = df_long[(df_long['state_t1'] == state_t1) & (df_long['state_t2'] == state_t2)]
    
    # Sort by the absolute change in frequency
    sorted_genes = transition_df.sort_values(by='change_in_frequency', key=abs, ascending=False)
    
    # Select the top 5 genes
    top_genes = sorted_genes.head(5)
    
    # Store the results
    top_genes_by_transition[(state_t1, state_t2)] = top_genes[['gene', 'change_in_frequency']]

# Print results
for transition, genes in top_genes_by_transition.items():
    print(f"Top genes for transition {transition}:")
    print(genes)
    print()
