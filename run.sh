
CUDA_VISIBLE_DEVICES=2,3,4,5 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 4

# CUDA_VISIBLE_DEVICES=2 python3 projects/CAViT/train.py --config-file projects/CAViT/configs/cavit_prid2011.yml  --num-gpus 1  --eval-only MODEL.WEIGHTS logs/11.113/base4_shift-att-v12_prid2011/model_best.pth TEMP.TEST.ALL True
