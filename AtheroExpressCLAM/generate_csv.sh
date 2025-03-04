#!/bin/bash

conda activate /hpc/local/Rocky8/dhl_ec/software/mambaforge3/envs/convocals
CSV_INPUT_DATA="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/20231004.CONVOCALS.samplelist.withSMAslides.csv"
CSV_DATA="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/AtheroExpress_HE_WSI_dataset_binary_IPH_new.csv"
CSV_DATA_BAL="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/AtheroExpress_HE_WSI_dataset_binary_IPH_eq_new.csv"
H5_DIR="/hpc/dhl_ec/VirtualSlides/HE/PROCESSED/features_imagenet/h5_files"
CLASSIFIER="IPH.bin"

python generate_label_updated.py --csv_input $CSV_INPUT_DATA \
--csv_output $CSV_DATA \
--csv_output_bal $CSV_DATA_BAL \
--h5_dir $H5_DIR \
--classifier $CLASSIFIER