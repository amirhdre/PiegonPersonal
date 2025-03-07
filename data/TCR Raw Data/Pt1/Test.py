import os
import pandas as pd

def load_tcr_data(file_path):
    """Load TCR data from a file."""
    df = pd.read_csv(file_path, sep='\t')  # Assuming tab-separated files
    file_name = os.path.basename(file_path).split('.')[0]
    sample_id, patient_id, state = file_name.split('_')[0], file_name.split('_')[1], file_name.split('_')[-1]  # Extract sample, patient, and state (F or R)
    df['sample'] = sample_id
    df['patient'] = patient_id
    df['state'] = state
    # Rearranging columns to make 'sample', 'patient', and 'state' the leftmost
    return df[['sample', 'patient', 'state', 'v', 'd', 'j', 'freq']]

def compute_frequency_change(patient_folder):
    """Compute change in frequency of V, D, and J genes within a patient."""
    files = sorted(
        [f for f in os.listdir(patient_folder) if f.endswith('.txt') and not f.startswith('metadata')],
        key=lambda x: int(x.split('_')[0])
    )
    
    if len(files) < 2:
        return pd.DataFrame()
    
    all_genes = set()
    
    for f in files:
        df = load_tcr_data(os.path.join(patient_folder, f))
        all_genes.update(df[['v', 'd', 'j']].apply(lambda x: '_'.join(x.dropna()), axis=1))
    
    all_genes = sorted(all_genes)
    transitions = []
    
    for i in range(len(files) - 1):
        file1, file2 = files[i], files[i + 1]
        path1, path2 = os.path.join(patient_folder, file1), os.path.join(patient_folder, file2)
        df1, df2 = load_tcr_data(path1), load_tcr_data(path2)
        
        patient_id = df1['patient'].iloc[0]
        sample1, sample2 = df1['sample'].iloc[0], df2['sample'].iloc[0]
        state1, state2 = df1['state'].iloc[0], df2['state'].iloc[0]
        
        freq1 = df1.set_index(df1[['v', 'd', 'j']].apply(lambda x: '_'.join(x.dropna()), axis=1))['freq'].to_dict()
        freq2 = df2.set_index(df2[['v', 'd', 'j']].apply(lambda x: '_'.join(x.dropna()), axis=1))['freq'].to_dict()
        
        freq_change = {gene: freq2.get(gene, 0) - freq1.get(gene, 0) for gene in all_genes}
        freq_change.update({
            'patient': patient_id,
            'sample_t1': sample1, 'state_t1': state1,
            'sample_t2': sample2, 'state_t2': state2,
            'file_t1': file1, 'file_t2': file2
        })
        
        transitions.append(freq_change)
    
    # Create DataFrame from transitions and reorder columns
    df_changes = pd.DataFrame(transitions)
    
    # Reorder the columns to make patient and file-related columns the leftmost
    patient_and_file_columns = ['patient', 'sample_t1', 'state_t1', 'sample_t2', 'state_t2', 'file_t1', 'file_t2']
    remaining_columns = [col for col in df_changes.columns if col not in patient_and_file_columns]
    
    return df_changes[patient_and_file_columns + remaining_columns]

def main():
    base_dir = '/Users/matthewfarah/pigeon/data/TCR Raw Data'
    all_changes = []
    
    for patient in os.listdir(base_dir):
        patient_folder = os.path.join(base_dir, patient)
        if os.path.isdir(patient_folder):
            df_changes = compute_frequency_change(patient_folder)
            if not df_changes.empty:
                all_changes.append(df_changes)
    
    final_df = pd.concat(all_changes, ignore_index=True) if all_changes else pd.DataFrame()
    final_df.to_csv('tcr_frequency_changes.csv', index=False)
    print("Saved frequency changes to tcr_frequency_changes.csv")

if __name__ == "__main__":
    main()
