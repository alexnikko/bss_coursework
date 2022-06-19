import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler

from src.models import TasNet, DualRNNmodel, SepformerWrapper, SuDORMRF
from src.utils import read_and_clean_meta, prepare_dataset_meta
from src.transforms import Normalizer
from src.datasets import VoxcelebDataset
from src.losses import Snr, Sdr, SiSnr, SiSdr, Pit

SR = 16_000  # noqa
num_spk = 2


def build_conv_tasnet() -> TasNet:
    model_params = dict(
        enc_dim=512,
        feature_dim=128,
        sr=SR,
        win=2,
        layer=8,
        stack=3,
        kernel=3,
        num_spk=num_spk,
        causal=False
    )
    model = TasNet(**model_params)
    return model


def build_dual_path_rnn() -> DualRNNmodel:
    model_params = dict(
        in_channels=512,
        out_channels=128,
        hidden_channels=160,
        kernel_size=2,
        rnn_type='LSTM',
        norm='gln',
        dropout=0.05,
        bidirectional=True,
        num_layers=6,
        k=SR * 15,
        num_spks=num_spk
    )
    model = DualRNNmodel(**model_params)
    return model


def build_sepformer() -> SepformerWrapper:
    model_params = dict(
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=128,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_norm='gln',
        masknet_useextralinearlayer=False,
        masknet_extraskipconnection=True,
        masknet_numspks=2,
        intra_numlayers=6,
        inter_numlayers=6,
        intra_nhead=8,
        inter_nhead=8,
        intra_dffn=512,
        inter_dffn=512,
        intra_use_positional=True,
        inter_use_positional=True,
        intra_norm_before=True,
        inter_norm_before=True,
    )
    model = SepformerWrapper(**model_params)
    return model


def build_sudo_rm_rf() -> SuDORMRF:
    model_params = dict(
        out_channels=256,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2
    )
    model = SuDORMRF(**model_params)
    return model


def build_model(model_name: str) -> nn.Module:
    """
    :param model_name:
        must be conv_tasnet, dual_path_rnn, sepformer or sudo_rm_rf
    :return:
        built model
    """
    if model_name == 'conv_tasnet':
        return build_conv_tasnet()
    elif model_name == 'dual_path_rnn':
        return build_dual_path_rnn()
    elif model_name == 'sepformer':
        return build_sepformer()
    elif model_name == 'sudo_rm_rf':
        return build_sudo_rm_rf()
    else:
        raise Exception(f'model_name must be equal to one of: '
                        f'conv_tasnet, dual_path_rnn, sepformer or sudo_rm_rf\n'
                        f'but got model_name={model_name}')


def build_criterion(criterion_name: str) -> nn.Module:
    """
    :param criterion_name:
        must be one of: SNR, SDR, SI-SNR, SI-SDR
    :return:
        nn.Module - PIT criterion to minimize
    """
    if criterion_name == 'SNR':
        sep_criterion = Snr()
    elif criterion_name == 'SDR':
        sep_criterion = Sdr()
    elif criterion_name == 'SI-SNR':
        sep_criterion = SiSnr()
    elif criterion_name == 'SI-SDR':
        sep_criterion = SiSdr()
    else:
        raise Exception(f'criterion_name must be equal to one of: '
                        f'SNR, SDR, SI-SNR, SI-SDR \n'
                        f'but got criterion_name={criterion_name}')
    criterion = Pit(
        metric_func=sep_criterion,
        eval_func='min'
    )
    return criterion


def build_metric_dict(metrics: list[str]) -> dict[str, nn.Module]:
    metric_dict = {}
    metric_to_func = {
        'SNR': Snr(upper_bound=30, negative=False),
        'SDR': Sdr(upper_bound=30, negative=False),
        'SI-SNR': SiSnr(upper_bound=30, negative=False),
        'SI-SDR': SiSdr(upper_bound=30, negative=False)
    }
    for available_metric, metric_func in metric_to_func.items():
        if available_metric in metrics:
            metric_dict[available_metric] = metric_func
    if not metric_dict:
        raise Exception(f'Pass at least one of possible metrics')
    return metric_dict


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


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    optimizer_params = dict(
        lr=1e-4
    )
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    opt_lr = optimizer.param_groups[0]['lr']
    scheduler_params = dict(
        init_lr=opt_lr / 10,
        peak_lr=opt_lr,
        warmup_steps=warmup_steps,
    )
    scheduler = WarmupLRScheduler(
        optimizer=optimizer,
        **scheduler_params
    )
    return scheduler


if __name__ == '__main__':
    model_names = ['conv_tasnet', 'dual_path_rnn', 'sepformer', 'sudo_rm_rf']
    for model_name in model_names:
        model = build_model(model_name)
        n_params = sum(p.numel() for p in model.parameters()) / 10 ** 6
        print(f'{model_name} has {n_params} M. parameters')
