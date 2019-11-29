FILE=$1
if [ $FILE == "celeba" ]; then # taken from stargan https://github.com/yunjey/stargan
    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./datasets/celeba.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE

elif [ $FILE == "selfie2anime" ]; then # dataset from https://github.com/taki0112/UGATIT
    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/9lz6gwwwyyxpdnn/selfie2anime.zip?dl=0
    ZIP_FILE=./datasets/selfie2anime.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE

fi
