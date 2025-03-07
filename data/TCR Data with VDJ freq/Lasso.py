from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the data
df = pd.read_csv("gene_frequency_changes_by_transition.csv", index_col=0)

# Identify metadata and gene columns
metadata_cols = ['patient', 'sample', 'state_t1', 'state_t2', 'transition']
gene_cols = [col for col in df.columns if col not in metadata_cols]

# Encode the transition (target) variable
df['transition_encoded'] = df['transition'].astype('category').cat.codes

# Extract features (X) and target variable (y)
X = df[gene_cols]  # Gene frequency change columns
y = df['transition_encoded']  # Encoded transition labels

# Handle missing values (impute)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply Lasso Logistic Regression (L1 regularization)
lasso_log_reg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
lasso_log_reg.fit(X_train, y_train)

# Coefficients for the features
coef = lasso_log_reg.coef_

# Get the important genes (those with non-zero coefficients)
important_genes = pd.DataFrame({
    'Gene': gene_cols,
    'Coefficient': coef.flatten()
})

# Filter for non-zero coefficients (important features)
important_genes = important_genes[important_genes['Coefficient'] != 0]

# Sort by the absolute value of the coefficients
important_genes['Abs_Coefficient'] = important_genes['Coefficient'].abs()
important_genes = important_genes.sort_values(by='Abs_Coefficient', ascending=False)

# Display the top important genes
print(important_genes.head(10))
