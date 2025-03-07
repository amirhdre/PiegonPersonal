import pandas as pd
df=pd.read_csv("tcr_frequency_changes.csv")
# Strip any leading/trailing spaces from column names
df['transition'] = df['state_t1'] + '->' + df['state_t2']

# List of gene columns to calculate the standard deviation for
gene_columns = [
    'TRBV10-1_TRBD1_TRBJ2-1', 'TRBV10-1_TRBD1_TRBJ2-5', 'TRBV10-1_TRBD1_TRBJ2-7', 
    'TRBV10-1_TRBD2, TRBD1_TRBJ2-5', 'TRBV10-1_TRBD2, TRBD1_TRBJ2-7', 'TRBV10-1_TRBD2_TRBJ1-1', 
    'TRBV10-1_TRBD2_TRBJ2-1', 'TRBV10-1_TRBD2_TRBJ2-5', 'TRBV10-1_TRBD2_TRBJ2-7', 'TRBV10-1_TRBJ2-3'
]

# Group by transition type and calculate standard deviation for each gene column
std_devs = df.groupby('transition')[gene_columns].std()
std_devs.to_csv("std_devs.csv")
# Display the results
print(std_devs)
