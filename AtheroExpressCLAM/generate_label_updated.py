import pandas as pd
import os
import argparse

# Setup parser for command line arguments
parser = argparse.ArgumentParser(description='Process WSI datasets')
parser.add_argument('--csv_input', type=str, default="/path/to/input.csv", help='input csv file name including path')
parser.add_argument('--csv_output', type=str, default="/path/to/output.csv", help='output csv file name including path')
parser.add_argument('--csv_output_bal', type=str, default="/path/to/output_balanced.csv", help='balanced output csv file name including path')
parser.add_argument('--h5_dir', type=str, default="/path/to/h5_files", help='path to h5 files')
parser.add_argument('--classifier', type=str, default="IPH.bin", help='name of variable to use as classifier')
args = parser.parse_args()

# Load the source CSV file
df_source = pd.read_csv(args.csv_input)
print(df_source.head())

# Directory containing the files
all_files = set(os.listdir(args.h5_dir))

# Lists to store the final data
slide_ids = []
final_labels = []
case_ids = []

# Iterate over all files in the directory
for file in all_files:
    if '.h5' in file:
        file_prefix = file.split('.', 1)[0]  # Split on the first dot
        study_num = file_prefix[2:]  # Exclude "AE" prefix
#        print('study_num:', study_num)
        # Get the label for the corresponding STUDY_NUMBER
        label_row = df_source[df_source['STUDY_NUMBER'].astype(str) == study_num]
#        print('label_row:', label_row)
        if not label_row.empty:
            label = label_row[args.classifier].values[0]
#            print('label', label)
            if pd.notna(label) and label != "NA":
                case_id = f"AE{study_num}"
                slide_id = file.rsplit('.', 1)[0]  # Remove file extension
                slide_ids.append(slide_id)
                final_labels.append(label)
                case_ids.append(case_id)

# Create the final DataFrame
df_output = pd.DataFrame({
    "case_id": case_ids,
    "slide_id": slide_ids,
    "label": final_labels
})

# Save the dataframe to the target CSV file
df_output.to_csv(args.csv_output, index=False)

# Balancing the dataset
dataset_unbalanced = df_output

# Count cases per category
count_cat_yes = dataset_unbalanced['label'].value_counts(dropna=False).get('yes', 0)
count_cat_no = dataset_unbalanced['label'].value_counts(dropna=False).get('no', 0)

smaller_count = min(count_cat_yes, count_cat_no)

# Split dataframe into two dataframes
df_yes = dataset_unbalanced[dataset_unbalanced['label'] == 'yes']
df_no = dataset_unbalanced[dataset_unbalanced['label'] == 'no']

# Randomly sample n rows from each dataframe
sample_df = pd.concat([df_yes.sample(smaller_count), df_no.sample(smaller_count)]).sample(frac=1)

sample_df.to_csv(args.csv_output_bal, index=False, sep=',')
