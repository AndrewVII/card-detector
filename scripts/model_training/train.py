from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os
from detectron2 import model_zoo
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    # Path to your dataset directory
    dataset_dir = "dataset/annotations"
    output_dir = "./output"
    os.system(f"rm -rf {output_dir}")

    # model
    model_name = os.getenv("MODEL")

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
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.RETINANET.NUM_CLASSES = 1  # Number of classes in your dataset
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.OUTPUT_DIR = output_dir

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate the model (optional)
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("my_dataset", cfg, False, output_dir=f"{output_dir}/")
    val_loader = build_detection_test_loader(cfg, "my_dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)
