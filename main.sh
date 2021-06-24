#!/usr/bin/env bash

FOLDERS=$( ls ./data/dataset | awk '{ print $1}' )

# echo $FOLDERS
FOLDER_PLUS_FILE=()
FILES=()

for folder in $FOLDERS
do
    FILES=$(ls ./data/dataset/$folder |  grep -i "_single.nii"  )  
    for file in $FILES
    do
        FOLDER_PLUS_FILE+=($folder'/'$file)
    done
done
 
REF_IMAGE=${FOLDER_PLUS_FILE[0]} 

for INPUT_IMAGE in ${FOLDER_PLUS_FILE[@]}
do
    echo -e "\nREFERENCE:" "./data/dataset/"$REF_IMAGE  "\nINPUT:" "./data/dataset/$INPUT_IMAGE"  "\nOUTPUT:" + "./data/output/dataset/"$INPUT_IMAGE
done




# fsl5.0-flirt \
#  -in ./data/TCGA_CS_4942_19970222/TCGA_CS_4942_19970222_post-contrast.nii \
#  -ref ./data/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_post-contrast.nii \
#  -out ./Output/OUTPUT \
#  -omat ./Output/OUTPUT.mat \
#  -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear

