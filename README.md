# Council-GAN
Implementation of our paper Breaking the Cycle - Colleagues are all you need (CVPR 2020)
### Paper
[Ori Nizan](https://onr.github.io/) , [Ayellet Tal](http://webee.technion.ac.il/~ayellet/),
**[Breaking the Cycle - Colleagues are all you need](https://arxiv.org/abs/1911.10538 "Breaking the cycle -- Colleagues are all you need")**
**[[Project](https://onr.github.io/Council_web/)]**

![gan_council_teaser](/images/paper_teaser.png)

![gan_council_overview](/images/gan_council_overview.png)

![male2female_gif](/images/m2f_gif.gif)

![glasses_gif](/images/glasses_gif.gif)

![anime_gif](/images/anime_gif.gif)

#### Temporary Telegram Bot
Send image to this [telegram bot](https://t.me/Council_GAN_bot) and it will send back its female translation

## Usage
### Install requirements

    pip install -r pip_requirements.txt
    conda env create -f conda_requirements.yml

### Downloading the dataset
#### Download the selfie to anime dataset:

    bash ./scripts/download.sh U_GAT_IT_selfie2anime

#### Download the celeba glasses removal dataset:

    bash ./scripts/download.sh celeba_glasses_removal

#### Download the celeba male to female dataset:

    bash ./scripts/download.sh celeba_male2female
#### use your on dataset:
```
├──datasets
    └──DATASET_NAME
        ├──testA
            ├──im1.png
            ├──im2.png
            └── ...
        ├──testB
            ├──im3.png
            ├──im4.png
            └── ...
        ├──trainA
            ├──im5.png
            ├──im6.png
            └── ...
        └──trainB
            ├──im7.png
            ├──im8.png
            └── ...
```
and change the **data_root** attribute to **./datasets/DATASET_NAME** in the yaml file

### Training:
#### Selfie to anime:
    python train.py --config configs/anime2face_council_folder.yaml --output_path ./outputs/council_anime2face_256_256 --resume

#### Glasses removel:
    python train.py --config configs/galsses_council_folder.yaml --output_path ./outputs/council_glasses_128_128 --resume

#### Male to female:
    python train.py --config configs/male2female_council_folder.yaml --output_path ./outputs/male2famle_256_256 --resume


### Testing:
for converting all the images in input_folder using all the members in the council:

    python test_on_folder.py --config configs/anime2face_council_folder.yaml --output_folder ./outputs/council_anime2face_256_256 --checkpoint ./outputs/council_anime2face_256_256/anime2face_council_folder/checkpoints/01000000 --input_folder ./datasets/selfie2anime/testB --a2b 0

or using spsified memeber:

    python test_on_folder.py --config configs/anime2face_council_folder.yaml --output_folder ./outputs/council_anime2face_256_256 --checkpoint ./outputs/council_anime2face_256_256/anime2face_council_folder/checkpoints/b2a_gen_3_01000000.pt --input_folder ./datasets/selfie2anime/testB --a2b 0
 
### Download Pretrain Models

#### Download pretrain male to female model:

    bash ./scripts/download.sh pretrain_male_to_female
    
#### Download pretrain glasses removal model:

    bash ./scripts/download.sh pretrain_glasses_removal
    
#### Download pretrain selfie to anime model:

    bash ./scripts/download.sh pretrain_selfie_to_anime



### Test GUI:
![gan_council_overview](/images/test_gui.png)

#### test GUI on pretrain model:

    python test_gui.py --config pretrain/m2f/128/male2female_council_folder.yaml --checkpoint pretrain/m2f/128/a2b_gen_0_01000000.pt --a2b 1

#### Citation
```
@inproceedings{nizan2020council,
  title={Breaking the Cycle - Colleagues are all you need},
  author={Ori Nizan, Ayellet Tal},
  booktitle={IEEE conference on computer vision and pattern recognition (CVPR)},
  year={2020}
}
```
