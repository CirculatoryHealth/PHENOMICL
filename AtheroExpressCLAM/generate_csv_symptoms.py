import pandas as pd
import numpy as np

# Load the CSV files
file1_path = '/hpc/dhl_ec/VirtualSlides/EVG/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_IPH.csv'
file2_path = '/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/20231004.CONVOCALS.samplelist.withSMAslides.csv'

file1_df = pd.read_csv(file1_path)
file2_df = pd.read_csv(file2_path)

# Filter the second dataframe based on the Symptoms.Update2G column
filtered_file2_df = file2_df[file2_df['Symptoms.Update2G'].notna()].copy()

# Create the label column based on Symptoms.Update2G values
filtered_file2_df['label'] = filtered_file2_df['Symptoms.Update2G'].apply(lambda x: 'yes' if x == 'Symptomatic' else 'no')

# Merge the filtered second dataframe with the first dataframe to match on case_id
# First, we need to ensure that STUDY_NUMBER from the second file matches with case_id in the first file
# We will create a new case_id column in the second dataframe for merging purposes
filtered_file2_df['case_id'] = 'AE' + filtered_file2_df['STUDY_NUMBER'].astype(str)

# Now merge based on case_id
merged_df = pd.merge(filtered_file2_df[['case_id', 'label']], file1_df[['case_id', 'slide_id']], on='case_id')

# Reorder columns so that label is the third column
merged_df = merged_df[['case_id', 'slide_id', 'label']]

# Save the generated CSV file with the label column in the third position
output_file_path = '/hpc/dhl_ec/VirtualSlides/EVG/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_2gsymptom.csv'
merged_df.to_csv(output_file_path, index=False)

# Now create a balanced dataset
yes_df = merged_df[merged_df['label'] == 'yes']
no_df = merged_df[merged_df['label'] == 'no']

# Determine the smaller group size
min_size = min(len(yes_df), len(no_df))

# Randomly sample from both groups to ensure equal amounts of 'yes' and 'no'
balanced_yes_df = yes_df.sample(min_size, random_state=42)
balanced_no_df = no_df.sample(min_size, random_state=42)

# Concatenate the two dataframes to create a balanced dataset
balanced_df = pd.concat([balanced_yes_df, balanced_no_df])

# Shuffle the rows to mix 'yes' and 'no' labels
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_output_file_path = '/hpc/dhl_ec/VirtualSlides/EVG/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_2gsymptom_eq.csv'
balanced_df.to_csv(balanced_output_file_path, index=False)
