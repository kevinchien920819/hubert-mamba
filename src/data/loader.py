from torch.utils.data import DataLoader

from config.hubert_mamba import HubertMambaConfig

from .dataset import HubertPretrainDataset
from .sampler import TokenBatchSampler


def get_hubert_dataloader(cfg: HubertMambaConfig, split: str, shuffle: bool = True) -> DataLoader:
    dataset = HubertPretrainDataset(
        manifest_dir=cfg.data.manifest_dir,
        label_dir=cfg.data.label_dir,
        split=split,
        label_name=cfg.data.labels[0],
        sample_rate=cfg.data.sample_rate,
        label_rate=cfg.data.label_rate,
        max_sample_size=cfg.data.max_sample_size,
        min_sample_size=cfg.data.min_sample_size,
        pad_audio=cfg.data.pad_audio,
        random_crop=cfg.data.random_crop,
        normalize=cfg.data.normalize,
    )

    max_tokens = int(cfg.dataloader.token_batch_size)
    if max_tokens > 0:
        batch_sampler = TokenBatchSampler(
            lengths=dataset.get_lengths(),
            max_tokens=max_tokens,
            shuffle=shuffle,
            drop_last=False,
            seed=cfg.general.seed,
        )
        return DataLoader(
            dataset,
            num_workers=cfg.dataloader.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=dataset.collate_fn,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
            prefetch_factor=cfg.dataloader.prefetch_factor if cfg.dataloader.num_workers > 0 else None,
        )

    subset_key = "dev" if split == cfg.data.valid_split else "train"
    return DataLoader(
        dataset,
        num_workers=cfg.dataloader.num_workers,
        batch_size=cfg.dataloader.batch_size.get(subset_key, cfg.dataloader.batch_size["train"]),
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
        prefetch_factor=cfg.dataloader.prefetch_factor if cfg.dataloader.num_workers > 0 else None,
    )
