# CAViT

[ECCV 2022] CAViT: Contextual Alignment Vision Transformer for Video Object Re-identification


## Preparation
- datasets
  - MARS
  - MARS\_DL
  - LS_VID
  - PRID-2011
  - iLIDS-VID
  - vveri901

- add data path
run `export FASTREID_DATASETS=/path/to/datasets/` in the terminal,
or,
add `export FASTREID_DATASETS=/path/to/datasets/` to your `~/.bashrc`

- install requirement
```
conda create -n reid python=3.7
conda activate reid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch

# if you use A100 gpu, please install pytorch >= 1.7 and cudatoolkit >=11.0
# for example
# conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r docs/requirements.txt
```


## Support backbone:
- swin transformer \& swin transformer 3D
- ViT \& ViT 3D (timeformer)
- TSM
- AP3D
- BickNet
- token shift





## Training
1. If you want to train with 4-GPU, run:
```
CUDA_VISIBLE_DEVICES=2 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 1

``` 

2. If you want to train with 4-GPU, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 4  HEADS.NORM syncBN

``` 

## Testing
1. If you want to test with 1-GPU, run:
```
CUDA_VISIBLE_DEVICES=2 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 1  --eval-only MODEL.WEIGHTS logs/prid2011/model_best.pth TEMP.TEST.ALL True
```

2. If you want to test with 4-GPUs, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 4  --eval-only MODEL.WEIGHTS logs/prid2011/model_best.pth TEMP.TEST.ALL True  HEADS.NORM syncBN
```


This repo also implements the fixed length sequeence testing \& the flexible length testing.
1. Flexible testing (use all frames in the sequence for testing)
```
CUDA_VISIBLE_DEVICES=2 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 1  --eval-only MODEL.WEIGHTS logs/prid2011/model_best.pth TEMP.TEST.ALL True
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 4  --eval-only MODEL.WEIGHTS logs/prid2011/model_best.pth TEMP.TEST.ALL True  HEADS.NORM syncBN
```


2. Fixed length testing

```
CUDA_VISIBLE_DEVICES=2 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 1  --eval-only MODEL.WEIGHTS logs/prid2011/model_best.pth TEMP.TEST.ALL False  TEMP.TEST.SEQ_SIZE  8   TEMP.TEST.TRACK_SPLIT 128
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 4  --eval-only MODEL.WEIGHTS logs/prid2011/model_best.pth HEADS.NORM syncBN TEMP.TEST.ALL False  TEMP.TEST.SEQ_SIZE  8   TEMP.TEST.TRACK_SPLIT 128
```


## Notice
- Dropout in ViT can have an impact on the results and lead to unstable experimental results,
- Our residual position embedding will influence the stable of experimental results. If you want to get more stable performance, please change `CASCADE.TPE` from `flow` to `sin`,
- PRID-2011 and iLIDS-VID are too samlle. The experimental results on these two data sets are unstable, particulary on PRID-2011. (e.g., Diffent GPUs, pytorch versions, and etc., all of them may influence the final results.)
- If you train models with multi-gpus, please fix `HEADS.NORM: BN` to `HEADS.NORM: syncBN` in the config file. This may influence the results.

## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: jinlin.wu@nlpr.ia.ac.cn


## ACKNOWLEDGEMENTS
The code was developed based on the ’fast-reid’ toolbox https://github.com/JDAI-CV/fast-reid.
