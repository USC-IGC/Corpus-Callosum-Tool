#!/bin/bash
#$ -S /bin/bash

# Input directory of T1
dirI=./CC/input

# Subject list for T1's to be processed
dirS=./CC/subject_list.txt

# Model folder path (github)
dirM=./CC/model

# Output directory 
dirO=./CC/output
mkdir -p ${dirO}/processed_input
mkdir -p ${dirO}/segmentation
mkdir -p ${dirO}/metrics
mkdir -p ${dirO}/QC

#########################################################################################################################
########## DO NOT CHANGE THIS PATH ########## 
# Python Script path 
script=./CC/scripts
cd ${script}
#########################################################################################################################

# Step 1: Preprocessing T1------> This step can be run on the CPU, required fsl-5.0.9
sh t1_norm.sh -i ${dirI} -m ${dirM} -s ${dirS} -o ${dirO}/processed_input

#########################################################################################################################

# Step 2: Generating CC segmentation------> This step needs to be run only on GPU
export PATH=./Anaconda3/envs/cc_pipe_seg/bin:$PATH
python generate_segmentations.py --inp ${dirO}/processed_input/T1_norm --model_path ${dirM} --out ${dirO}

#########################################################################################################################

# Step 3: Extract and Collate metrics-------> This step can be run on the CPU
export PATH=./Anaconda3/envs/cc_pipe_metric/bin:$PATH
for entry in "${dirO}/segmentation/nifti"/*
do
	python extract_metrics.py --mask ${entry} --output ${dirO}/metrics/${entry}.csv
	echo "Done with ${entry}"
done
python collate_metrics.py --inp ${dirO}/metrics --out ${dirO}

##########################################################################################################################

# Step 4: Auto QC for generated segmentations------> This step can be run on the CPU
export PATH=./Anaconda3/envs/cc_pipe_qc/bin:$PATH
python auto_qc.py --inp ${dirO} --model ${dirM} --out ${dirO}


