CUDA_PREPEND=0
CMD="python run_optim_demo.py
--model_type=dpt_large
--input_path=./input/coco_demo
--output_path=./precomputed_data/coco_demo/
--input=input/coco_demo/*.jpg
--mode=smpl_reproj
--dataset=demo
--config-file
./external/panoptic_deeplab/tools_d2/configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml
--opts
MODEL.WEIGHTS
./external/panoptic_deeplab/tools_d2/checkpoints/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.pth
"

echo CUDA_VISIBLE_DEVICES=$CUDA_PREPEND $CMD
CUDA_VISIBLE_DEVICES=$CUDA_PREPEND $CMD