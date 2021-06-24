#!/usr/bin/env bash

BASE_FOLDER="./data/dataset"
OUTPUT_FOLDER="./data/output"
FILE_SUFFIX="post-contrast.nii"

FOLDERS=$(ls $BASE_FOLDER | awk '{ print $1}')

# echo $FOLDERS
FOLDER_PLUS_FILE=()
FILES=() 

for folder in ${FOLDERS[@]}
do
    FILES=$(ls $BASE_FOLDER/$folder |  grep -i $FILE_SUFFIX)  
    for file in $FILES
    do
        FOLDER_PLUS_FILE+=($folder/$file)
        REF_IMAGE_FOLDER=$folder
    done
 

mkdir -p $OUTPUT_FOLDER/$REF_IMAGE_FOLDER 

done

echo -e "" > ./data/output/log.txt

COUNTER=0 

for INPUT_IMAGE in ${FOLDER_PLUS_FILE[@]}
do 
    REF_IMAGE=${FOLDER_PLUS_FILE[0]} 
    let COUNTER=$COUNTER+1
    echo "Progressing File "$( echo $INPUT_IMAGE)"... Complete! "

    echo -e "\nREFERENCE:" $BASE_FOLDER/$REF_IMAGE  \
    "\nINPUT:" "$BASE_FOLDER/$INPUT_IMAGE"          \
    "\nOUTPUT:" "$OUTPUT_FOLDER/$INPUT_IMAGE"       \
    >> ./data/output/log.txt

    $(${FSLDIR}/bin/fsl5.0-flirt \
        -in $BASE_FOLDER/$INPUT_IMAGE                   \
        -ref $BASE_FOLDER/$REF_IMAGE                    \
        -out $OUTPUT_FOLDER/$INPUT_IMAGE                \
        -omat $OUTPUT_FOLDER/$INPUT_IMAGE               \
        -bins 256 -cost corratio -searchrx -90 90       \
        -searchry -90 90 -searchrz -90 90 -dof 12       \
        -interp trilinear 
    )

done





