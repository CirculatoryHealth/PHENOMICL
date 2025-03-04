import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Process WSI datasets')
# Arguments
parser.add_argument('--csv_input', type=str, 
                    default="/hpc/dhl_ec/VirtualSlides/CD34/20231004.CONVOCALS.samplelist.withSMAslides.csv",
                    help='input csv file name including path')
parser.add_argument('--csv_output', type=str, 
                    default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/dataset_csv/AtheroExpress_CD34_WSI_dataset_binary_IPH.csv",
                    help='output csv file name including path')
parser.add_argument('--csv_output_bal', type=str, 
                    default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/dataset_csv/AtheroExpress_CD34_WSI_dataset_binary_IPH_eq.csv",
                    help='balanced output csv file name including path')
parser.add_argument('--h5_dir', type=str, 
                    default="/hpc/dhl_ec/VirtualSlides/CD34/PROCESSED/features_imagenet/h5_files", 
                    help='path to h5 files')
parser.add_argument('--classifier', type=str, default="IPH.bin", 
                    help='name of variable to use as classifier')
args = parser.parse_args()

# Load the source CSV file
df_source = pd.read_csv(args.csv_input)

# Directory containing the files
all_files = set(os.listdir(args.h5_dir))

# Lists to store the final data
slide_ids = []
final_labels = []
case_ids = []

# Set to keep track of the files we've matched
matched_files = set()

# Iterate over the rows of the source dataframe
for _, row in df_source.iterrows():
    study_num = str(row["STUDY_NUMBER"])
    label = row[args.classifier]
    
    # Check for matching files
    matching_files = [file for file in all_files if study_num in file and file not in matched_files]
    
    if not matching_files:
        print(f"No match for STUDY_NUMBER: {study_num}")
        continue  # move to the next iteration if no match

    # If there's a match and label is valid, add to the final lists
    if pd.notna(label) and label != "NA":
        matched_file = matching_files[0]
        slide_id = ".".join(matched_file.split(".")[:-1])
        slide_ids.append(slide_id)
        final_labels.append(label)
        case_ids.append(slide_id.split(".")[0])
        
        # Mark this file as matched
        matched_files.add(matched_file)

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
count_cat_yes = dataset_unbalanced['label'].value_counts(dropna=False)['yes']
count_cat_no = dataset_unbalanced['label'].value_counts(dropna=False)['no']

smaller_count = min(count_cat_yes, count_cat_no)

# Split dataframe into two dataframes
df_yes = dataset_unbalanced[dataset_unbalanced['label'] == 'yes']
df_no = dataset_unbalanced[dataset_unbalanced['label'] == 'no']

# Randomly sample n rows from each dataframe
sample_df = pd.concat([df_yes.sample(smaller_count), df_no.sample(smaller_count)]).sample(frac=1)

sample_df.to_csv(args.csv_output_bal, index=False, sep=',')

