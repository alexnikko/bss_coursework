from src.utils import read_and_clean_meta, prepare_dataset_meta
from src.transforms import Normalizer
from src.datasets import VoxcelebDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    meta_root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb'
    id2gender = read_and_clean_meta(meta_root)

    root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb/voxceleb1/test/wav'
    minimum_duration = 3  # in seconds
    male_speakers, female_speakers, sp2files = prepare_dataset_meta(root, id2gender, minimum_duration=minimum_duration)

    SR = 16_000
    frames = minimum_duration * SR
    steps = 100
    prob_same = 0.5
    mean_level_db = -26
    std_level_db = 3
    transform = Normalizer(mean=mean_level_db, std=std_level_db)

    dataset = VoxcelebDataset(
        root=root,
        male_speakers=male_speakers,
        female_speakers=female_speakers,
        sp2files=sp2files,
        frames=frames,
        steps=steps,
        prob_same=prob_same,
        transform=transform,
    )

    batch_size = 8
    num_workers = 2
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    for batch in loader:
        mix, src = batch
        print(mix.shape)
        print(src.shape)
        break
