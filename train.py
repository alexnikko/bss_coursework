import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from src.utils import read_and_clean_meta, prepare_dataset_meta
from src.transforms import Normalizer
from src.datasets import VoxcelebDataset
from src.models.conv_tasnet import TasNet
from src.models.dual_path_rnn import DualRNNmodel
from src.models.sepformer import SepformerWrapper
from src.models.sudormrf import SuDORMRF
from src.losses import Pit, Snr

if __name__ == '__main__':
    meta_root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb'
    id2gender = read_and_clean_meta(meta_root)

    root = '/Users/alexnikko/prog/bss_coursework/datasets/voxceleb/voxceleb1/test/wav'
    minimum_duration = 1  # in seconds
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

    batch_size = 1
    num_workers = 2
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    for batch in loader:
        mix, src = batch
        print(mix.shape)
        print(src.shape)
        break
    # net = TasNet()
    # net = DualRNNmodel(256, 64, 128, bidirectional=True, norm='ln', num_layers=6)
    # net = SepformerWrapper(
    #     encoder_kernel_size=16,
    #     encoder_in_nchannels=1,
    #     encoder_out_nchannels=256,
    #     masknet_chunksize=250,
    #     masknet_numlayers=2,
    #     masknet_norm="ln",
    #     masknet_useextralinearlayer=False,
    #     masknet_extraskipconnection=True,
    #     masknet_numspks=2,
    #     intra_numlayers=8,
    #     inter_numlayers=8,
    #     intra_nhead=8,
    #     inter_nhead=8,
    #     intra_dffn=1024,
    #     inter_dffn=1024,
    #     intra_use_positional=True,
    #     inter_use_positional=True,
    #     intra_norm_before=True,
    #     inter_norm_before=True,
    # )
    net = SuDORMRF(out_channels=128,
                   in_channels=512,
                   num_blocks=16,
                   upsampling_depth=4,
                   enc_kernel_size=21,
                   enc_num_basis=512,
                   num_sources=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = Pit(Snr(20), eval_func='min')
    losses = []
    n_steps = 1000
    mix, src = next(iter(loader))
    for _ in tqdm(range(n_steps)):
        optimizer.zero_grad()
        output = net(mix)
        loss = loss_fn(output, src)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(losses[-1])
