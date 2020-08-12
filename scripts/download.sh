#!/usr/bin/env bash
FILE=$1

#celeba dataset
# @inproceedings{liu2015faceattributes,
# title = {Deep Learning Face Attributes in the Wild},
# author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
# booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
# month = {December},
# year = {2015}
#}
if [ $FILE == "celeba_male2female" ]; then
    URL=https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_male2female.zip?dl=0
    ZIP_FILE=./datasets/celeba_male2female.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE

elif [ $FILE == "celeba_glasses_removal" ]; then
    URL=https://cgm.technion.ac.il/Computer-Graphics-Multimedia/CouncilGAN/DataSet/celeba_glasses.zip?dl=0
    ZIP_FILE=./datasets/celeba_glasses.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE

#  selfie2anime dataset used in https://github.com/taki0112/UGATIT
#  title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
#  author={Kim, Junho and Kim, Minjae and Kang, Hyeonwoo and Lee, Kwanghee},
#  journal={arXiv preprint arXiv:1907.10830},
#  year={2019}
elif [ $FILE == "U_GAT_IT_selfie2anime" ]; then
  URL=https://www.dropbox.com/s/9lz6gwwwyyxpdnn/selfie2anime.zip?dl=0
  ZIP_FILE=./datasets/selfie2anime.zip
  mkdir -p ./datasets/
  wget -N $URL -O $ZIP_FILE
  unzip $ZIP_FILE -d ./datasets/
  rm $ZIP_FILE

elif [ $FILE == "pretrain_male_to_female" ]; then
    URL="https://onedrive.live.com/download?cid=552378614E6BA583&resid=552378614E6BA583%2119549&authkey=ANzlMZsBQy77urU"
    ZIP_FILE=./pretrain/pretrain_m2f.zip

    mkdir -p ./pretrain/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrain/
    rm $ZIP_FILE

elif [ $FILE == "pretrain_glasses_removal" ]; then
    URL="https://onedrive.live.com/download?cid=552378614E6BA583&resid=552378614E6BA583%2119556&authkey=AAjSgID8ia6_Wt8"
    ZIP_FILE=./pretrain/pretrain_glasses_removal.zip
    mkdir -p ./pretrain/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrain/
    rm $ZIP_FILE

elif [ $FILE == "pretrain_selfie_to_anime" ]; then
    URL="https://onedrive.live.com/download?cid=552378614E6BA583&resid=552378614E6BA583%2119566&authkey=AIcUuinR_eQPp1c"
    ZIP_FILE=./pretrain/pretrain_anime.zip
    mkdir -p ./pretrain/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrain/
    rm $ZIP_FILE


else
    echo "Available arguments are celeba_male2female, celeba_glasses_removal, U_GAT_IT_selfie2anime, pretrain_male_to_female, pretrain_glasses_removal, pretrain_selfie_to_anime."
    exit 1

fi
