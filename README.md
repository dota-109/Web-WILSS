# Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation. 
## Chang Liu, Giulia Rizzoli, Pietro Zanuttigh, Fu Li, Yi Niu -- ECCV 2024

#### Official PyTorch Implementation
![headfig](https://github.com/dota-109/Web-WILSS/blob/main/docs/head-fig.png)

Current weakly-supervised incremental learning for semantic segmentation (WILSS) approaches only consider replacing pixel-level annotations with image-level labels, while the training images are still from well-designed datasets. In this work, we argue that widely available web images can also be considered for the learning of new classes. To achieve this, firstly we introduce a strategy to select web images which are similar to previously seen examples in the latent space using a Fourier-based domain discriminator.  Then, an effective caption-driven reharsal strategy is proposed to preserve previously learnt classes. To our knowledge, this is the first work to rely solely on web images for both the learning of new concepts and the preservation of the already learned ones in WILSS. Experimental results show that the proposed approach can reach state-of-the-art performances without using manually selected and annotated data in the incremental steps. 

![method](https://github.com/dota-109/Web-WILSS/blob/main/docs/main-framework.png)

## How to run
the code is based on the [WILSON](https://github.com/fcdl94/WILSON).
### Requirements
We have simple requirements:
The main requirements are:
```
python > 3.1
pytorch > 1.6
```
If you want to install a custom environment for this code, you can run the following using [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html):
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install tensorboard
conda install jupyter
conda install matplotlib
conda install tqdm
conda install imageio

pip install inplace-abn # this should be done using CUDA compiler (same version as pytorch)
pip install wandb # to use the WandB logger
```

### Datasets 
To download dataset, follow the scripts: `data/download_voc.sh`, `data/download_coco.sh` 

If your datasets are in a different folder, make a soft-link from the target dataset to the data folder.
We expect the following tree:
```
data/voc/
    SegmentationClassAug/
        <Image-ID>.png
    JPEGImages/
        <Image-ID>.png
    split/
    ... other files 
```
:warning: Be sure not to override the current `voc` directory of the repository. 
We suggest to link the folders inside the voc directory.

### Web Data :bowtie:
We use pascal web data to preserve previous knowledge and class web data for learning new knowledge. They have been uploaded to Baidu Drive:

**Pascal web data**
```
https://pan.baidu.com/s/1W19rDo9CWUU4_zJ3pFrSeQ?pwd=1234
pswd: 1234
```
**Class web data**
```
https://pan.baidu.com/s/1UeiL8cWjQvtWN68N2g00TA?pwd=1234 
pswd: 1234
```

**Usage**  
For the usage of pascal web data, 1)first we generate the new caption based on those web data, 2)then a cosine similarity score is computed for selecting web images that are more similar to the original one. 
3)Once finish the filtering, "generate_label" file and "generate_train_file" file would help us to train main model with selected web data:  
**Process**  
1): We offer our caption files generated by LLM openflamingo, you could find it in "captions" folder.    
2): By running "filter_replay_with_NN" and "move_with_filtered_caption" function in "utils/measure_noun.py" respectively, you will get the selected caption and image data.  
3): Run "generate_label.py" to get the pseudo annotation for web data and run "generate_train_file.py" for generating the path file.  
4): Set "--replay_path" to your web data path and now you can train your model with web data.  

***Selected web data for 10-10 and 15-5***  
In case you have trouble in filtering web data, we also provide our selected data so that you can directly train your model without any filtering strategy.   

15-5-ov:  
[google drive](https://drive.google.com/file/d/1337cglt-abKHcGmEnfg3y4QuhRdwm3H8/view?usp=drive_link)

10-10-ov:  
[google drive](https://drive.google.com/file/d/1kmNIsWspDrWYSJT8L8llifboTen23T8S/view?usp=drive_link)

15-5s-ov:  
[google drive](https://drive.google.com/file/d/16TVotGvSQ8mfXv7aUPgA6ipp52pESFMA/view?usp=drive_link)

10-10s-ov:  
[google drive](https://drive.google.com/file/d/1eiESTddf0Vc9KgpYS4r7avi3QJ-O2EvR/view?usp=drive_link)

coco:  
[google drive](https://drive.google.com/file/d/1bh-J2cXG0rSRa6aWGsLnmcoHdJ0x_M4J/view?usp=drive_link)

*Usage*: Download the data, unizp and set the"--replay_path" to the path just unzipped.

### ImageNet Pretrained Models
After setting the dataset, you download the models pretrained on ImageNet using [InPlaceABN](https://github.com/mapillary/inplace_abn).
[Download](https://drive.google.com/file/d/1rQd-NoZuCsGZ7_l_X9GO1GGiXeXHE8CT/view) the ResNet-101 model (we only need it but you can also [download other networks](https://github.com/mapillary/inplace_abn) if you want to change it).
Then, put the pretrained model in the `pretrained` folder.

### Run
We provide different an example script to run the experiments (see `run.sh`).
In the following, we describe the basic parameter to run an experiment.
First, we assume that we have a command 
```
exp='python -m torch.distributed.launch --nproc_per_node=<num GPUs> --master_port <PORT> run.py --num_workers <N_Workers>'`
```
that allow us to setup the distributed data parallel script.

The first to replicate us, is to obtain the model on the step 0 (base step, fully supervised). You can run:
```
exp --name Base --step 0 --lr 0.01 --bce --dataset <dataset> --task <task> --batch_size 24 --epochs 30 --val_interval 2 [--overlap]
```
where we use `--bce` to train the classifier with the binary cross-entropy. `dataset` can be `voc` or `coco-voc`. The task 
are, 
```
voc: (you can set overlap here)
    15-5, 10-10
coco: (overlap is not used)
    voc 
```

After this, you can run the incremental steps using only image level labels (set the `weakly` parameter).
```
exp --name ours --step 1 --weakly --lr 0.001 --alpha 0.5 --step_ckpt <pretr> --loss_de 1 --lr_policy warmup --affinity \ 
    --dataset <dataset> --task <task> --batch_size 24 --epochs 40 [--overlap] --replay --replay_path <your replay data path> --replay_num 2
```
where `pretr` should be the path to the pretrained model (usually `checkpoints/step/<dataset>-<task>/<name>.pth`).
