# Corpus-Callosum-Tool: 
## Segmentation, quality control and abnormality classification of the midsagittal corpus callosum

This automated pipeline can be used for accurate Corpus Callosum (CC) segmentation across multiple MRI datasets and extract a variety of features to describe the shape of the CC. We also include an automatic quality control function to detect poor segmentations using Machine Learning.

## How to use the tool:
* Clone the github directory using: git clone https://github.com/USC-IGC/Corpus-Callosum-Tool.git
* Create three different virtual environments using the packages mentioned in "packages" folder.
* In run_CC.sh file:
  * Once the virtual environments are installed, add the python paths for segmentation, metrics extraction and auto QC in line 34, 40 and 51 respectively.
  * Input: Apply bias field correction (eg: ANTs N4) on T1's.
  * Put all the bias field corrected T1's in one folder and put the path for the same on line 5.
  * Create a text file with all the subject id's of the T1's to be processed and put the path to the text file on line 8.
  * Add the model directory on line 11.
  * Set the output path folder where all the results would be generated.
* All the steps can be run on CPU.
* The final output will be "metrics_qc.csv" in the output folder which will have all the metrics and a column "QC label" indicating whether the segmentations were accurate(0)/fail(1).


## If you use this code, please cite the following paper:
### Gadewar, Shruti P., Elnaz Nourollahimoghadam, Ravi R. Bhatt, Abhinaav Ramesh, Shayan Javid, Iyad Ba Gari, Alyssa H. Zhu, Sophia Thomopoulos, Paul M. Thompson, and Neda Jahanshad. "A Comprehensive Corpus Callosum Segmentation Tool for Detecting Callosal Abnormalities and Genetic Associations from Multi Contrast MRIs." ArXiv (2023): arXiv-2305. (https://arxiv.org/abs/2305.01107)
