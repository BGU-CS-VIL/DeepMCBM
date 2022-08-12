# DeepMCBM

**Authors:** Guy Erez, Ron Shapira Weber, and Oren Freifeld.

This code repository corresponds to our ECCV '22 paper: **DeepMCBM: A Deep Moving-camera Background Model**.
DeepMCBM is a novel 2D-based method for unsupervised learning of a moving-camera background model, which is highly scalable and allows for relatively-free camera motion.

## DeepMCBM Examples
https://user-images.githubusercontent.com/6692232/180310568-def4a578-091e-4a51-98c7-036e3f76f1cc.mp4

https://user-images.githubusercontent.com/6692232/180310715-9ba0d7c1-7075-476f-98e9-a964b56beadf.mp4

https://user-images.githubusercontent.com/6692232/180310721-69e822fb-89e4-46d4-89ac-fb1d9a7fc6b4.mp4

https://user-images.githubusercontent.com/6692232/180310726-bbb9a9ed-60fd-4774-8e27-22bbec92db9b.mp4

https://user-images.githubusercontent.com/6692232/180310729-1aafeeb5-36fa-4622-85aa-96e14c26c245.mp4

https://user-images.githubusercontent.com/6692232/180310734-79522a80-47ab-4391-8339-927953fdf779.mp4

https://user-images.githubusercontent.com/6692232/180310703-9390b353-37eb-41ca-802f-7ba4ffa42abd.mp4

# Documentation:    

## Environment
The repository is equipped with a DeepMCBM_env.yml file.  
Run conda env create -f DeepMCBM_env.yml from your terminal to set a conda environment using this file.     
To ensure the environment is set properly, activate the new environment and run a "dry run" with few epochs:
```
conda activate DeepMCBM
python src/DeepMCBM.py --DryRun
```
## Train, Predict and Evaluate  
To train, predict and evaluate a deepMCBM module on the default tennis sequence:
```
python src/DeepMCBM.py 
```
## Input, Output and Checkpoints 
The default values for the input, output, and checkpoints paths are set in src/args.py and can be changed to any path you wish. The requirement for the input directory is to have the following subdirectories: "frames" include the sequence frames, and if ground truth labels are available, a "GT" directory containing the ground truth frames. See the input/tennis for an example. The output directories are named by the sequence and Neptune.ai experiment: output/sequence_name/experiment_number in this directory you will find:
- background_estimation directory containing the background estimation of the model.  
- MSE directory containing the Mean Square Error (MSE) computed using the ground truth labels. 
- panoramic_robust_mean.png image, shows the alignment result of the STN module. 

## Predict a Pretrained Model
To only predict and evaluate metrics:
```
python src/DeepMCBM.py --no_train_BMN --no_train_STN 
```
You can change the loaded checkpoint using a flag: 
```
python src/DeepMCBM.py --no_train_BMN --no_train_STN --BMN_ckpt ckpt_file.ckpt  
```
Or by editing the MCBM_CKPT argument in src/args.py  

Note: when using a pretrained model, the argument --pad, describing the size of the padding, must be the same as in the training phase.

Visual Comparisons: Coming soon

