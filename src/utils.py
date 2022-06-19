import pandas as pd
import soundfile as sf
import os
from glob import glob
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Optional


def read_and_clean_meta(root: str, cache: Optional[str] = None) -> dict[str, str]:
    if cache is not None and os.path.exists(cache):
        return pd.read_pickle(cache)
    meta_vox1 = pd.read_csv(os.path.join(root, 'vox1_meta.csv'), sep='\t')
    meta_vox2 = pd.read_csv(os.path.join(root, 'vox2_meta.csv'), sep='\t')
    for meta in [meta_vox1, meta_vox2]:
        meta.columns = [col.strip() for col in meta.columns]
        for col in meta:
            meta[col] = meta[col].str.strip()
    meta_vox1['from'] = 'vox1'
    meta_vox2['from'] = 'vox2'
    meta_vox1.rename(columns={'VoxCeleb1 ID': 'voxceleb_id'})
    meta_vox2.rename(columns={'VoxCeleb2 ID': 'voxceleb_id'})
    meta = pd.concat((meta_vox1, meta_vox2)).reset_index()
    meta.to_csv(os.path.join(root, 'meta.csv'), index=False)
    id2gender = dict(zip(meta['VoxCeleb1 ID'], meta['Gender']))
    if cache is not None:
        pd.to_pickle(id2gender, cache)
    return id2gender


def get_speaker_files(root: str, speaker: str) -> list[str]:
    speaker_root = os.path.join(root, speaker)
    files = glob(f'{speaker_root}/**/*.wav', recursive=True)
    files = [file.removeprefix(f'{speaker_root}/') for file in files]
    return files


def get_file_duration_info(root: str, speaker: str, speaker_file: str) -> tuple[str, int, float]:
    filepath = os.path.join(root, speaker, speaker_file)
    a, sr = sf.read(filepath)
    n_frames = len(a)
    duration = n_frames / sr
    return filepath, n_frames, duration


def prepare_dataset_meta(root: str, id2gender: dict[str, str], minimum_duration: float,
                         cache: Optional[str] = None):
    if cache is not None and os.path.exists(cache):
        return pd.read_pickle(cache)
    speakers = os.listdir(root)
    # because it is macOS
    if '.DS_Store' in speakers:
        speakers.remove('.DS_Store')

    sp2files = defaultdict(list)
    for speaker in tqdm(speakers, desc='Iterating over speakers'):
        speaker_files = get_speaker_files(root, speaker)
        for file in tqdm(speaker_files, desc=f'Iterating over {speaker} files'):
            _, n_frames, duration = get_file_duration_info(root, speaker, file)
            if duration < minimum_duration:
                continue
            sp2files[speaker].append({
                'rel_path': file,
                'n_frames': n_frames
            })
    male_speakers = [speaker for speaker in speakers if id2gender[speaker] == 'm']
    female_speakers = [speaker for speaker in speakers if id2gender[speaker] == 'f']
    if cache is not None:
        pd.to_pickle((male_speakers, female_speakers, sp2files), cache)
    return male_speakers, female_speakers, sp2files


if __name__ == '__main__':
    root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb'
    id2gender = read_and_clean_meta(root)
    root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb/voxceleb1/test/wav'
    male_speakers, female_speakers, sp2files = prepare_dataset_meta(root, id2gender, minimum_duration=3)
    print(len(male_speakers), len(female_speakers), len(sp2files), sep='\n')
    