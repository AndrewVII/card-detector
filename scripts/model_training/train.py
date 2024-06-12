import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.data.datasets import register_coco_instances

dataset_dir = "dataset/annotations"
register_coco_instances(
    "my_dataset",
    {},
    os.path.join(dataset_dir, "result.json"),
    os.path.join(dataset_dir, "images"),
)

dataset_metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")
