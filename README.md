# GAN-Council
implementation of our paper Breaking the Cycle - Colleagues are all you need 
### Paper
[Ori Nizan](https://onr.github.io/) , [Ayellet Tal](http://webee.technion.ac.il/~ayellet/),
**[Breaking the Cycle - Colleagues are all you need](https://arxiv.org/abs/1911.10538 "Breaking the cycle -- Colleagues are all you need")**

![gan_council_teaser](/images/paper_teaser.png)

![gan_council_overview](/images/gan_council_overview.png)

![glasses_gif](/images/glasses_gif.gif)

![male2female_gif](/images/m2f_gif.gif)

![anime_gif](/images/anime_gif.gif)


## Usage

### Downloading the dataset
#### download the selfie to anime dataset:

    bash ./scripts/download.sh U_GAT_IT_selfie2anime

#### download the celeba glasses removal dataset:

    bash ./scripts/download.sh celeba_glasses_removal
    
#### download the celeba male to female dataset:

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
#### selfie to anime: 
    python train.py --config configs/anime2face_council_folder.yaml --output_path ./outputs/council_anime2face_256_256 --resume 

#### glasses removel:
    python train.py --config configs/galsses_council_folder.yaml --output_path ./outputs/council_glasses_128_128 --resume 
    
#### male to female:
    python train.py --config configs/male2female_council_folder.yaml --output_path ./outputs/male2famle_256_256 --resume 


### Testing:
for converting all the images in input_folder using all the members in the council:

    python test_on_folder.py --config configs/anime2face_council_folder.yaml --output_folder ./outputs/council_anime2face_256_256 --checkpoint ./outputs/council_anime2face_256_256/anime2face_council_folder/checkpoints/01000000 --input_folder ./datasets/selfie2anime/testB --b2a
    
or using spsified memeber:

    python test_on_folder.py --config configs/anime2face_council_folder.yaml --output_folder ./outputs/council_anime2face_256_256 --checkpoint ./outputs/council_anime2face_256_256/anime2face_council_folder/checkpoints/b2a_gen_3_01000000.pt --input_folder ./datasets/selfie2anime/testB --b2a
        
#### Test GUI:
![gan_council_overview](/images/test_gui.png)

    python test_gui.py --config configs/galsses_council_folder.yaml --checkpoint ./outputs/council_glasses_128_128/galsses_council_folder/checkpoints/a2b_gen_0_00700000.pt
    
#### Citation
