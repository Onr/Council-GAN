#!/usr/bin/env bash
FILE=$1
if [ $FILE == "celeba_male2female" ]; then # taken from stargan https://github.com/yunjey/stargan
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./datasets/celeba.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE

elif [ $FILE == "celeba_glasses_removal" ]; then # taken from stargan https://github.com/yunjey/stargan
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./datasets/celeba.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE

elif [ $FILE == "selfie2anime" ]; then # dataset from https://github.com/taki0112/UGATIT
    URL=https://www.dropbox.com/s/9lz6gwwwyyxpdnn/selfie2anime.zip?dl=0
    ZIP_FILE=./datasets/selfie2anime.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE
else
    echo "Available arguments are celeba_male2female, celeba_glasses_removal, selfie2anime."
    exit 1


fi
