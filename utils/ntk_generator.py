# Reference: https://github.com/zalanborsos/bilevel_coresets
import numpy as np
from jax import jit
from neural_tangents import stax


_, _, kernel_fn = stax.serial(
    stax.Dense(100, 1., 0.05),
    stax.Relu(),
    stax.Dense(100, 1., 0.05),
    stax.Relu(),
    stax.Dense(10, 1., 0.05))
fnn_kernel_fn = jit(kernel_fn, static_argnums=(2,))


def generate_fnn_ntk(X, Y):
    return np.array(fnn_kernel_fn(X, Y, 'ntk'))


def ResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
        channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, Shortcut),
                       stax.FanInSum())


def ResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [ResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [ResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def Resnet(block_size, num_classes):
    return stax.serial(
        stax.Conv(64, (3, 3), padding='SAME'),
        ResnetGroup(block_size, 64),
        ResnetGroup(block_size, 128, (2, 2)),
        ResnetGroup(block_size, 256, (2, 2)),
        ResnetGroup(block_size, 512, (2, 2)),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.05))


_, _, resnet_kernel_fn = Resnet(block_size=2, num_classes=10)
resnet_kernel_fn = jit(resnet_kernel_fn, static_argnums=(2,))


def generate_resnet_ntk(X, Y, skip=25):
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(0, m, skip):
        K[:, i:i + skip] = np.array(resnet_kernel_fn(X, Y[i:i + skip], 'ntk'))
    return K / 10


def get_kernel_fn(bone):
    from backbone.MNISTMLP import MNISTMLP
    from backbone.ResNet18 import ResNet

    if isinstance(bone, MNISTMLP):
        return lambda x, y: generate_fnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))
    elif isinstance(bone, ResNet):
        return lambda x, y: generate_resnet_ntk(x.transpose(0, 2, 3, 1), y.transpose(0, 2, 3, 1))
    else:
        raise NotImplementedError('Neural Tangent Kernel is not implemented for this backbone')
