import pandas as pd
import os

# Load the source CSV file
source_csv = "/hpc/dhl_ec/VirtualSlides/CD34/20231004.CONVOCALS.samplelist.withSMAslides.csv"
df_source = pd.read_csv(source_csv)

# Directory containing the files
h5_folder = "/hpc/dhl_ec/VirtualSlides/CD34/PROCESSED/features_imagenet/h5_files"
all_files = set(os.listdir(h5_folder))

# Lists to store the final data
slide_ids = []
final_labels = []
case_ids = []

# Set to keep track of the files we've matched
matched_files = set()

# Iterate over the rows of the source dataframe
for _, row in df_source.iterrows():
    study_num = str(row["STUDY_NUMBER"])
    label = row["IPH.bin"]
    
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
target_csv = "/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/dataset_csv/AtheroExpress_CD34_WSI_dataset_binary_IPH.csv"
df_output.to_csv(target_csv, index=False)

# create balanced datasets
dataset_unbalanced = df_output

# count cases per category
count_cat_yes = dataset_unbalanced['label'].value_counts(dropna=False)['yes']
count_cat_no = dataset_unbalanced['label'].value_counts(dropna=False)['no']

if count_cat_yes < count_cat_no:
    smaller_count = count_cat_yes
else:
    smaller_count = count_cat_no

# split dataframe into two dataframes
df_yes = dataset_unbalanced[dataset_unbalanced['label'] == 'yes']
df_no = dataset_unbalanced[dataset_unbalanced['label'] == 'no']

# randomly sample n rows from each dataframe
sample_df = pd.concat([df_yes.sample(smaller_count), df_no.sample(smaller_count)])

# randomize the rows of the resulting dataframe
sample_df = sample_df.sample(frac=1)

sample_df.to_csv("/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/dataset_csv/AtheroExpress_CD34_WSI_dataset_binary_IPH_eq.csv", index=False, sep=',')
