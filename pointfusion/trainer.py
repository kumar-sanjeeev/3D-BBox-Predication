import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from pointfusion.datasets.SereactDataset import SereactDataset
from pointfusion.models.pointfusion import PointFusionLit


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    # Load Processed Dataset
    dataset = SereactDataset(root_path=cfg.dataset.processed_data_path)

    # Calculate the sizes for train and validation sets
    total_size = len(dataset)
    val_size = int(cfg.dataset.val_split_ratio * total_size)
    test_size = int(cfg.dataset.test_split_ratio * total_size)
    train_size = total_size - val_size - test_size

    # Split the dataset
    seed = torch.Generator().manual_seed(cfg.dataset.seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=seed
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )
    val_dataloader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=cfg.dataset.num_workers
    )
    test_dataloader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=cfg.dataset.num_workers
    )

    print(f"Total dataset size: {len(dataset)}")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    batch = next(iter(train_dataloader))

    images = batch["img_t"]
    pcs = batch["pc_t"]
    target = batch["bbox_t"]

    print(f"Images: {images.size()}")
    print(f"Point Cloud: {pcs.size()}")
    print(f"Target: {target.size()}")

    # Create the model
    model = PointFusionLit(
        num_points=cfg.model.num_points,
        fusion_type=cfg.model.fusion_type,
        lr=cfg.model.learning_rate,
        draw_bbox=cfg.model.draw_bbox,
    )

    # Pass the data module to the Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=cfg.trainer.logger,
        deterministic=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
