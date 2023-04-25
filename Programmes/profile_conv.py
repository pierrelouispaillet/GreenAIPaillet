import time
from functools import partial
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F


def compl_mul(a, b):
    """
    Given a and b two tensors of dimension 4
    with the last dimension being the real and imaginary part, 
    returns a multiplied by the conjugate of b, the multiplication
    being with respect to the second dimension.
    """
    op = partial(torch.einsum, "bct,dct->bdt")
    return torch.stack([
        op(a[..., 0], b[..., 0]) + op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) - op(a[..., 0], b[..., 1])
    ],
                       dim=-1)


class FastConv(nn.Module):
    """
    Convoluton based on FFT, faster for large kernels and small strides.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=True):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1))
        else:
            self.bias = None
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size))

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, signal):
        padded = F.pad(self.weight,
                       (0, signal.size(-1) - self.weight.size(-1)))
        signal_fr = torch.rfft(signal, 1)
        weight_fr = torch.rfft(padded, 1)
        output_fr = compl_mul(signal_fr, weight_fr)
        output = torch.irfft(output_fr, 1, signal_sizes=(signal.size(-1), ))
        output = output[..., ::self.stride]
        target_length = (signal.size(-1) - self.kernel_size) // self.stride + 1
        output = output[..., :target_length].contiguous()
        if self.bias is not None:
            output += self.bias
        return output


def profile(module, *args, repetitions=10, warmup=1):
    """
    Given a module and args, apply repeatedly the module to the args,
    calling `torch.cuda.synchronize()` in between. Return the time per
    repetition. Not perfect profiling but gives a rough idea.
    """
    module(*args)
    begin = time.time()
    for _ in range(repetitions):
        module(*args)
        torch.cuda.synchronize()
    return (time.time() - begin) / repetitions


def human_seconds(seconds, display='.2f'):
    """
    Human readable string from a number of seconds.
    """
    value = seconds * 1e6
    ratios = [1e3, 1e3, 60, 60, 24]
    names = ['us', 'ms', 's', 'min', 'hrs', 'days']
    last = names.pop(0)
    for name, ratio in zip(names, ratios):
        if value / ratio < 0.3:
            break
        value /= ratio
        last = name
    return f"{format(value, display)} {last}"


def test_one(kernel_size=1024,
             channels=8,
             batch_size=16,
             length=16000 * 5,
             stride=1):
    print(f"Benchmark for kernel_size={kernel_size} "
          f"stride={stride} channels={channels}")
    device = "cuda"
    signal = torch.randn(batch_size, channels, length, device=device)
    conv = nn.Conv1d(
        channels, channels, kernel_size, stride=stride, bias=False).to(device)
    fft_conv = FastConv(
        channels, channels, kernel_size, stride=stride, bias=False).to(device)
    fft_conv.weight = conv.weight

    conv_output = conv(signal)
    fft_output = fft_conv(signal)
    error = torch.abs(conv_output - fft_output)
    print("\tMean error={:.2g}, max error={:.2g}".format(
        error.mean(), error.max()))
    torch.backends.cudnn.benchmark = False
    print("\tCudnn benchmark = False: {}".format(
        human_seconds(profile(conv, signal))))
    torch.backends.cudnn.benchmark = True
    print("\tCudnn benchmark = True: {}".format(
        human_seconds(profile(conv, signal))))
    print("\tFFT Conv: {}".format(human_seconds(profile(fft_conv, signal))))


def test():
    print("torch.backends.cudnn.is_available(): ",
          torch.backends.cudnn.is_available(),
          "\ntorch.backends.cudnn.version(): ", torch.backends.cudnn.version())
    grid = product([256, 1024, 2048], [64], [1, 16])
    for kernel_size, channels, stride in grid:
        test_one(kernel_size=kernel_size, channels=channels, stride=stride)


if __name__ == "__main__":
    test()