# Body Size and Depth Disambiguation in Multi-Person Reconstruction from Single Images

### [[Project]](http://www.iri.upc.edu/people/nugrinovic/depthsize/index.html)[ [Paper]](http://www.iri.upc.edu/people/nugrinovic/depthsize/paper.pdf) 

## Requirements
- Python (tested on 3.8)
- Pytorch (tested on 1.7)
## Installation

### Create env
```
git clone https://github.com/nicolasugrinovic/size_depth_disambiguation.git
cd size_depth_disambiguation
conda create -n sizedepth python=3.8
conda activate sizedepth
pip install -r requirements.txt
```
### Install Detectron2
Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) so it can be accesible from the conda environment. The fastest way to install is via `pip`.

### Install Panoptic DeepLab
Go to "external" folder:

`cd ./external`

and install in this folder the following:
- [Panoptic DeepLab](https://github.com/bowenc0221/panoptic-deeplab)

### Download Midas-v3 weights
We use [Midas-v3](https://github.com/isl-org/DPT) as the depth detector. There is no need
for installation as the code used is provided here. However, 
you need to download the model weights and place them on `./weights` folder. The 
weights to be donwloaded are *dpt_large-midas-2f21e586.pt*
```
mkdir ./weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
```
### Install pytorch3d
Install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
. Though the only function used is `load_obj`. You may use
any other library to load the meshes, just be aware of the 
vertex order at reading. 
## Demo
To run the demo, you first need to generate 
(or precompute) data which is then used by the 
optimization method. You need to generate initial pose/shape 
estimations and give them the correct format. 
Alternatevily, you can download and use 
[this data](https://drive.google.com/file/d/1rwWMkVtdOcxABOL5G96EA7tv-gkECqdB/view?usp=sharing).

To precompute the data run the following command:
```
python precompute_estimation_data.py --input_path=./input/coco_demo --output_path=./precomputed_data/coco_demo/ --mode=smpl_reproj --dataset=demo --model_type=dpt_large --config-file ./external/panoptic_deeplab/tools_d2/configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml --opts MODEL.WEIGHTS ./external/panoptic_deeplab/tools_d2/checkpoints/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.pth
```
Change `input_path` accordingly to where you have the initial input data. 
This can have any location as long as it has the following structure:

```
input
|-- data_name
    `-- img_name1.jpg
    `-- img_name1_TRANS_person0.obj
    .
    .
    .
    `-- img_name1_TRANS_personN.obj
    `-- img_name1_3djoints_0.json
    .
    .
    .
    
    `-- img_name1_3djoints_N.json
    `-- img_name2.jpg
    .
    .
    .
```
This will generate the data inside `./precomputed_data/coco_demo/` folder.

Finally, to run the optimization from our method execute 
the following command:

```
python run_optim_demo.py --model_type=dpt_large --input_path=./input/coco_demo --output_path=./output/coco_demo --input=input/coco_demo/*.jpg --mode=smpl_reproj --plane_scale=1.0 --n_iters=2000 --w_ordinal_loss=0 --w_reg_size=0.0 --config-file ./external/panoptic_deeplab/tools_d2/configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml --opts MODEL.WEIGHTS ./external/panoptic_deeplab/tools_d2/checkpoints/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.pth
```
Alternatively, you can run the scripts `precompute_data.sh` and
`run_demo.sh` found in the `scripts` folder

## Test/Eval
Run `eval.py`

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@inproceedings{ugrinovic2021body,
  title={Body Size and Depth Disambiguation in Multi-Person Reconstruction from Single Images},
  author={Ugrinovic, Nicolas and Ruiz, Adria and Agudo, Antonio and Sanfeliu, Alberto and Moreno-Noguer, Francesc},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={53--63},
  year={2021},
  organization={IEEE}
}
```


[comment]: <> (generate results or donwload from...)

[comment]: <> (generate:)

[comment]: <> (-results_baseline)

[comment]: <> (-results_ours)

[comment]: <> (-results_frankmocap)

[comment]: <> (-initials_crmh)

[comment]: <> (-initials_frankmocap)

[comment]: <> (eval )
