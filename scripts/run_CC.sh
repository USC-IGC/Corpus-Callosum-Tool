#!/bin/bash
#$ -S /bin/bash


# Python(virtual environment) with required libraries for generating segmentations, change the path
py_path_seg=./python

# Python(virtual environment) with required libraries for Metrics extraction, change the path
py_path_metric=./python

# Python(virtual environment) with required libraries for Auto QC of segmentations, change the path
py_path_qc=./python

# Input directory of T1, change the path
dirI=./input

# Subject list for T1's to be processed, change the path
dirS=./subject_list.txt

# Model folder path (github), wherever the git repo has been cloned
dirM=./model

# Output directory, change the path 
dirO=./output
mkdir -p ${dirO}/processed_input
mkdir -p ${dirO}/segmentation
mkdir -p ${dirO}/metrics
mkdir -p ${dirO}/QC

######################################################################################################################### 
# Scripts path to where the git repo has been cloned
script=./scripts
cd ${script}
#########################################################################################################################


# Step 1: Preprocessing T1------> This step can be run on the CPU, required fsl-5.0.9
sh t1_norm.sh -i ${dirI} -m ${dirM} -s ${dirS} -o ${dirO}/processed_input

#########################################################################################################################

# Step 2: Generating CC segmentation------> This step needs to be run only on GPU
${py_path_seg} generate_segmentations.py --inp ${dirO}/processed_input/T1_norm --model_path ${dirM} --out ${dirO}

#########################################################################################################################

# Step 3: Extract and Collate metrics-------> This step can be run on the CPU
for entry in "${dirO}/segmentation/nifti"/*
do
	${py_path_metric} extract_metrics.py --mask ${entry} --output ${dirO}/metrics/${entry: -41:7}.csv
	echo "Done with ${entry: -41:7}"
done
${py_path_metric} collate_metrics.py --inp ${dirO}/metrics --out ${dirO}

##########################################################################################################################

# Step 4: Auto QC for generated segmentations------> This step can be run on the CPU
${py_path_qc} auto_qc.py --inp ${dirO} --model ${dirM} --out ${dirO}






