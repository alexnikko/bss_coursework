import torch
from torch.utils.data import DataLoader

from src.utils import read_and_clean_meta, prepare_dataset_meta
from src.transforms import Normalizer
from src.datasets import VoxcelebDataset


def build_datasets() -> tuple[VoxcelebDataset, VoxcelebDataset]:
    meta_root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb'
    id2gender_cache = '/Users/alexnikko/prog/bss_coursework/cache/id2gender.p'
    id2gender = read_and_clean_meta(meta_root, cache=id2gender_cache)

    train_root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb/voxceleb1/test/wav'
    test_root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb/voxceleb1/test/wav'
    minimum_duration = 3  # in seconds
    train_dataset_meta_cache = '/Users/alexnikko/prog/bss_coursework/cache/train_3s.p'
    train_male_speakers, train_female_speakers, train_sp2files = prepare_dataset_meta(
        root=train_root,
        id2gender=id2gender,
        minimum_duration=minimum_duration,
        cache=train_dataset_meta_cache
    )
    test_dataset_meta_cache = '/Users/alexnikko/prog/bss_coursework/cache/test_3s.p'
    test_male_speakers, test_female_speakers, test_sp2files = prepare_dataset_meta(
        root=train_root,
        id2gender=id2gender,
        minimum_duration=minimum_duration,
        cache=test_dataset_meta_cache
    )

    SR = 16_000  # noqa
    frames = minimum_duration * SR
    train_steps = 100
    test_steps = 50
    prob_same = 0.5
    mean_level_db = -26
    std_level_db = 3
    transform = Normalizer(mean=mean_level_db, std=std_level_db)

    train_dataset = VoxcelebDataset(
        root=train_root,
        male_speakers=train_male_speakers,
        female_speakers=train_female_speakers,
        sp2files=train_sp2files,
        frames=frames,
        steps=train_steps,
        prob_same=prob_same,
        transform=transform,
    )

    test_dataset = VoxcelebDataset(
        root=test_root,
        male_speakers=test_male_speakers,
        female_speakers=test_female_speakers,
        sp2files=test_sp2files,
        frames=frames,
        steps=test_steps,
        prob_same=prob_same,
        transform=transform,
    )

    print(f'Number of speakers in:\nTRAIN = {len(train_dataset.sp2files)}\nTEST = {len(test_dataset.sp2files)}')

    return train_dataset, test_dataset


def build_loaders(train_dataset: VoxcelebDataset, test_dataset: VoxcelebDataset) -> tuple[DataLoader, DataLoader]:
    batch_size = 1
    num_workers = 4
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
