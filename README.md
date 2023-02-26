## DiffusionInst: Diffusion Model for Instance Segmentation

```
export DETECTRON2_DATASETS=/home/data
CUDA_VISIBLE_DEVICE=0 screen python main.py --config-file configs/res50.yaml 
```

![](results/arch.jpeg)

## Model Performance

 Method          | Mask AP (1 step) | Mask AP (4 step) 
-----------------|:----------------:|:----------------:
 COCO-val-Res50  |       37.3       |       37.5       
 COCO-val-Res101 |       41.0       |       41.1       
 COCO-val-Swin-B |       46.6       |       46.8       

![](results/visual.jpeg)