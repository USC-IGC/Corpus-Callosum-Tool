#!/bin/bash
#$ -S /bin/bash



while getopts i:m:s:o: option
do 
    case "${option}"
        in
        i)inpdir=${OPTARG};;
        m)modeldir=${OPTARG};;
        s)subjtxt=${OPTARG};;
        o)outbasedir=${OPTARG};;
    esac
done


SUBJECTS=(`cat ${subjtxt}`)
outdir="${outbasedir}/T1_norm"
maskdir="${outbasedir}/temp/brainmasks"
T12MNI="${outbasedir}/T1MNI_6p"


mkdir -p ${outdir}
mkdir -p ${maskdir}
mkdir -p ${T12MNI}
mkdir -p ${T12MNI}/matrices



for subject in ${SUBJECTS};
do

# Register to MNI template
flirt -in ${inpdir}/${subject}.nii.gz -ref ${modeldir}/MNI152_T1_1mm.nii.gz -out ${T12MNI}/${subject}_MNI_6p.nii.gz -dof 6 -cost mutualinfo -omat ${T12MNI}/matrices/${subject}_T1_2MNI_6p.xfm

# BET and normalization
/usr/local/fsl-5.0.9/bin/bet ${T12MNI}/${subject}_MNI_6p.nii.gz ${maskdir}/${subject}_MNI_6p.nii.gz -f 0.3 
/usr/local/fsl-5.0.9/bin/fslmaths ${maskdir}/${subject}_MNI_6p.nii.gz -inm 50 "${outdir}/${subject}_MNI_6p.nii.gz"

echo "Done with ${subject}"
done


