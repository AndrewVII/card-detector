import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os
from detectron2 import model_zoo

if __name__ == "__main__":
    # Path to your dataset directory
    dataset_dir = "dataset/annotations"

    # model
    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    # Register the dataset
    register_coco_instances(
        "my_dataset", {}, os.path.join(dataset_dir, "result.json"), dataset_dir
    )

    # Load metadata and dataset
    dataset_metadata = MetadataCatalog.get("my_dataset")
    dataset_dicts = DatasetCatalog.get("my_dataset")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.DATASETS.TRAIN = ("my_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Update this if you have more than one class

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate the model (optional)
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("my_dataset", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)
