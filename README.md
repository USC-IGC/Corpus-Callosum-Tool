# Corpus-Callosum-Tool: 
## Segmentation, quality control and abnormality classification of the midsagittal corpus callosum

This automated pipeline can be used for accurate Corpus Callosum (CC) segmentation across multiple MRI datasets and extract a variety of features to describe the shape of the CC. We also include an automatic quality control function to detect poor segmentations using Machine Learning.

## How to use the tool:
* Clone the github directory using: git clone https://github.com/USC-IGC/Corpus-Callosum-Tool.git
* Create three different virtual environments using the packages mentioned in "packages" folder.
* In run_CC.sh file:
  * Once the virtual environments are installed, add the python paths for segmentation, metrics extraction and auto QC in line 6, 9 and 12 respectively.
  * Input: Apply bias field correction (eg: ANTs N4) on T1's.
  * Put all the bias field corrected T1's in one folder and put the path for the same on line 15.
  * Create a text file with all the subject id's of the T1's to be processed and put the path to the text file on line 18.
  * Add the model directory on line 21.
  * Set the output path folder where all the results would be generated.
* Run all the steps one at a time and sequentially (Step 1,3 and 4 on CPU and 2 on GPU only).
* Before running Step 2, make sure to change the available node on line 11 in generate_segmentations.py file.
* The final output will be "metrics_qc.csv" in the output folder which will have all the metrics and a column "QC label" indicating whether the segmentations were accurate(0)/fail(1).

