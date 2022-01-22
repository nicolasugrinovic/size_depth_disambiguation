# Copyright (c) Facebook, Inc. and its affiliates.
import os.path as osp

EXTERNAL_DIRECTORY = "./external"

# Configurations for PointRend.
POINTREND_PATH = osp.join(EXTERNAL_DIRECTORY, "detectron2/projects/PointRend")
POINTREND_CONFIG = osp.join(
    POINTREND_PATH, "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
)
POINTREND_MODEL_WEIGHTS = (
    "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/"
    "164955410/model_final_3c3198.pkl"
)

# Configurations for BodyMocap.
BODY_MOCAP_PATH = osp.join(EXTERNAL_DIRECTORY, "frankmocap")
BODY_MOCAP_REGRESSOR_CKPT = osp.join(
    # BODY_MOCAP_PATH,
    "extra_data/body_module/pretrained_weights",
    "2020_05_31-00_50_43-best-51.749683916568756.pt",
)
BODY_MOCAP_SMPL_PATH = osp.join(
    # BODY_MOCAP_PATH,
    "extra_data/smpl"
)

# Configurations for PHOSA
FOCAL_LENGTH = 1.0
IMAGE_SIZE = 640
REND_SIZE = 256  # Size of target masks for silhouette loss.
BBOX_EXPANSION_FACTOR = 0.3  # Amount to pad the target masks for silhouette loss.
SMPL_FACES_PATH = "models/smpl_faces.npy"

# Mapping from class name to COCO contiguous id. You can double-check these using:
# >>> coco_metadata = MetadataCatalog.get("coco_2017_val")
# >>> coco_metadata.thing_classes
CLASS_ID_MAP = {
    "bat": 34,
    "bench": 13,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "laptop": 63,
    "skateboard": 36,
    "surfboard": 37,
    "tennis": 38,
}
MESH_MAP = {  # Class name -> list of paths to objs.
    "bicycle": ["models/meshes/bicycle_01.obj"]
}

# Dict[class_name: List[Tuple(path_to_parts_json, interaction_pairs_dict)]].
PART_LABELS = {
    "person": [("models/meshes/person_labels.json", {})],
    "bicycle": [
        (
            "models/meshes/bicycle_01_labels.json",
            {"seat": ["butt"], "handle": ["lhand", "rhand"]},
        )
    ],
}


DETECTRON17_TO_24 = [19,20,21,22,23,9,8,10,7,11,6,3,2,4,1,5,0]
SMPL_TO_CPM_14 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14] # note 8 is missing


female_fields = {
    'hips_distances': 0.204178,
    'biceps_dists': 0.258424,
    'arm_dists': 0.233562,
    'thigh_dists': 0.461708,
    'leg_dists': 0.323735,
}

male_fields = {
    'hips_distances': 0.185262,
    'biceps_dists': 0.258785,
    'arm_dists': 0.259390,
    'thigh_dists': 0.438981,
    'leg_dists': 0.372745,
}
