import torch
import torch.nn as nn
from torchmetrics.functional import (
    signal_noise_ratio,
    signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    scale_invariant_signal_distortion_ratio
)
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as stoi
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq
from torchmetrics import PermutationInvariantTraining as Pit

from typing import Optional


class Snr(nn.Module):
    def __init__(self, upper_bound: Optional[float] = None, negative: bool = True):
        super().__init__()
        self.upper_bound = upper_bound
        self.negative = negative

    def forward(self, pred, gt):
        snr = signal_noise_ratio(pred, gt)
        if self.negative:
            snr = -snr
        if self.upper_bound is not None:
            snr = torch.clip(snr, max=self.upper_bound)
        return snr


class Sdr(nn.Module):
    def __init__(self, upper_bound: Optional[float] = None, negative: bool = True):
        super().__init__()
        self.upper_bound = upper_bound
        self.negative = negative

    def forward(self, pred, gt):
        sdr = signal_distortion_ratio(pred, gt)
        if self.negative:
            sdr = -sdr
        if self.upper_bound is not None:
            sdr = torch.clip(sdr, max=self.upper_bound)
        return sdr


class SiSnr(nn.Module):
    def __init__(self, upper_bound: Optional[float] = None, negative: bool = True):
        super().__init__()
        self.upper_bound = upper_bound
        self.negative = negative

    def forward(self, pred, gt):
        si_snr = scale_invariant_signal_noise_ratio(pred, gt)
        if self.negative:
            si_snr = -si_snr
        if self.upper_bound is not None:
            si_snr = torch.clip(si_snr, max=self.upper_bound)
        return si_snr


class SiSdr(nn.Module):
    def __init__(self, upper_bound: Optional[float] = None, negative: bool = True):
        super().__init__()
        self.upper_bound = upper_bound
        self.negative = negative

    def forward(self, pred, gt):
        si_sdr = scale_invariant_signal_distortion_ratio(pred, gt)
        if self.negative:
            si_sdr = -si_sdr
        if self.upper_bound is not None:
            si_sdr = torch.clip(si_sdr, max=self.upper_bound)
        return si_sdr


class Pesq:
    def __init__(self, sr: int = 16_000):
        self.sr = sr

    def __call__(self, pred, gt):
        return pesq(pred, gt, fs=self.sr, mode='wb')


class Stoi:
    def __init__(self, sr: int = 16_000):
        self.sr = sr

    def __call__(self, pred, gt):
        return stoi(pred, gt, fs=self.sr)


if __name__ == '__main__':
    import numpy as np
    from src.transforms import Normalizer
    x = torch.rand(2, 48_000)
    noise = torch.rand(2, 48_000)

    speech_normalizer = Normalizer(mean=-25, std=0)
    noise_normalizer = Normalizer(mean=-35, std=0)

    x = torch.from_numpy(np.stack([speech_normalizer(x[i].detach().numpy()) for i in range(len(x))]))[None]
    noise = torch.from_numpy(np.stack([speech_normalizer(noise[i].detach().numpy()) for i in range(len(noise))]))[None]

    x.requires_grad = True

    mix = x + noise

    # model = torch.nn.Linear(100, 100)
    # preds = model(mix)
    preds = 2 * mix
    SNR = Snr(upper_bound=10)
    SDR = Sdr(upper_bound=10)
    SI_SNR = SiSnr(upper_bound=10)
    SI_SDR = SiSdr(upper_bound=10)
    PESQ = Pesq()
    STOI = Stoi()

    metrics = [SNR, SDR, SI_SNR, SI_SDR, PESQ, STOI]
    for metric in metrics:
        print(metric, metric(preds, x))
