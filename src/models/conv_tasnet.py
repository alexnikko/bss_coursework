import torch
import torch.nn as nn
from torch.autograd import Variable

from src.models.modules import TCN


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3,
                 kernel=3, num_spk=2, causal=False):
        super(TasNet, self).__init__()

        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim

        self.win = int(sr * win / 1000)
        self.stride = self.win // 2

        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal

        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)  # noqa

        # TCN separator
        self.TCN = TCN(self.enc_dim, self.enc_dim * self.num_spk, self.feature_dim, self.feature_dim * 4,
                       self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)  # noqa

    def pad_signal(self, signal):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if signal.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if signal.dim() == 2:
            signal = signal.unsqueeze(1)
        batch_size = signal.size(0)
        nsample = signal.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(signal.type())
            signal = torch.cat([signal, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(signal.type())
        signal = torch.cat([pad_aux, signal, pad_aux], 2)

        return signal, rest

    def forward(self, input_):

        # padding
        output, rest = self.pad_signal(input_)
        batch_size = output.size(0)

        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return output


if __name__ == "__main__":
    x1 = torch.rand(2, 48_000)
    x2 = torch.rand(2, 48_000)
    src = torch.stack((x1, x2))
    x = x1 + x2
    net = TasNet()
    output = net(x)
    print(output.shape)
