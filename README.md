# Body Size and Depth Disambiguation in Multi-Person Reconstruction from Single Images

### [[Project]](http://www.iri.upc.edu/people/nugrinovic/depthsize/index.html)[ [Paper]](http://www.iri.upc.edu/people/nugrinovic/depthsize/paper.pdf) 

## Code
- Demo
- Eval

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


## Demo

## Test/Eval


## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{ugrinovic2021sizedepth,
    Author = {Ugrinovic, Nicolas and Ruiz, Adria and Agudo, Antonio and Sanfeliu, Alberto and Moreno-Noguer, Francesc},
    Title = {Body Size and Depth Disambiguation in Multi-Person Reconstruction from Single Images},
    Year = {2021},
    booktitle = {3DV},
}
```
