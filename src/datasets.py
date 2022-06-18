import numpy as np
import soundfile as sf
import os
from torch.utils.data import Dataset
from typing import Union


class VoxcelebDataset(Dataset):
    def __init__(self,
                 root: str,
                 male_speakers: list[str],
                 female_speakers: list[str],
                 sp2files: dict[str, list[dict[str, Union[str, int]]]],
                 frames: int,
                 steps: int,
                 prob_same: float = 0.5,
                 transform=None):
        super().__init__()

        self.root = root
        self.male_speakers = male_speakers
        self.female_speakers = female_speakers
        self.sp2files = sp2files
        self.frames = frames
        self.steps = steps
        self.prob_same = prob_same
        self.transform = transform
        assert 0 <= prob_same <= 1, f'prob_same must be in [0, 1], got {prob_same}'
        self.list_of_choice_for_same = ['female', 'male']
        self.prob_of_choice_for_same = [
            len(female_speakers) / (len(female_speakers) + len(male_speakers)),
            len(male_speakers) / (len(female_speakers) + len(male_speakers)),
        ]

    def __len__(self):
        return self.steps

    def __getitem__(self, idx: int):
        if np.random.rand() < self.prob_same:
            speakers = (self.female_speakers
                        if np.random.choice(self.list_of_choice_for_same, p=self.prob_of_choice_for_same) == 'female'
                        else self.male_speakers)
            sp1, sp2 = np.random.choice(speakers, size=2, replace=False)
        else:
            sp1 = np.random.choice(self.male_speakers)
            sp2 = np.random.choice(self.female_speakers)

        file1 = np.random.choice(self.sp2files[sp1])
        file2 = np.random.choice(self.sp2files[sp2])

        rel_path1, rel_path2 = file1['rel_path'], file2['rel_path']
        n_frames1, n_frames2 = file1['n_frames'], file2['n_frames']

        start1 = np.random.randint(n_frames1 - self.frames + 1)
        start2 = np.random.randint(n_frames2 - self.frames + 1)

        path1, path2 = os.path.join(self.root, sp1, rel_path1), os.path.join(self.root, sp2, rel_path2)

        a1 = sf.read(path1, frames=self.frames, start=start1, dtype='float32')[0]
        a2 = sf.read(path2, frames=self.frames, start=start2, dtype='float32')[0]

        if self.transform is not None:
            a1 = self.transform(a1)
            a2 = self.transform(a2)

        mix = a1 + a2

        return mix, np.stack((a1, a2))
