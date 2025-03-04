import pandas as pd

he_df = pd.read_csv("/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/AtheroExpress_HE_WSI_dataset_binary_IPH_new.csv")
evg_df = pd.read_csv("/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/AtheroExpress_EVG_WSI_dataset_binary_IPH_new.csv")

# Debug: Inspect column names
print("HE DataFrame columns:", he_df.columns)
print("EVG DataFrame columns:", evg_df.columns)

# Ensure 'case_id' exists in both DataFrames
if 'case_id' not in he_df.columns or 'case_id' not in evg_df.columns:
    raise KeyError("Missing 'case_id' column in one or both DataFrames.")

# Merge datasets on 'case_id'
combined_df = pd.merge(he_df, evg_df, on='case_id', how='outer', suffixes=('_HE', '_EVG'))

# Check consistency of labels
label_mismatch = combined_df[combined_df['label_HE'] != combined_df['label_EVG']]

# Output results
# Save or use the combined DataFrame
combined_df.to_csv('combined_output.csv', index=False)
print("\nLabel Mismatches (if any):")
print(label_mismatch)
