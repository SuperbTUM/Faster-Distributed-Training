# resnet.py

"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Only applicable for stride=1 at this time
def unsqueeze_all(t):
    # Helper function to unsqueeze all the dimensions that we reduce over
    return t[None, :, None, None]


def convolution_backward(grad_out, X, weight, stride, padding):
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1), stride=stride, padding=padding).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight, stride=stride, padding=padding)
    if X.size() != grad_X.size():
        diff_xaxis = X.size(-2) - grad_X.size(-2)
        diff_yaxis = X.size(-1) - grad_X.size(-1)
        p2d = (diff_yaxis >> 1, diff_yaxis - (diff_yaxis >> 1), diff_xaxis >> 1, diff_xaxis - (diff_xaxis >> 1))
        grad_X = F.pad(grad_X, p2d, "constant", 0.)
    if weight.size() != grad_input.size():
        diff_xaxis = weight.size(-2) - grad_input.size(-2)
        diff_yaxis = weight.size(-1) - grad_input.size(-1)
        p2d = (diff_yaxis >> 1, diff_yaxis - (diff_yaxis >> 1), diff_xaxis >> 1, diff_xaxis - (diff_xaxis >> 1))
        grad_input = F.pad(grad_input, p2d, "constant", 0.)
    return grad_X, grad_input


def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # We use the formula: out = (X - mean(X)) / (sqrt(var(X)) + eps)
    # in batch norm 2d's forward. To simplify our derivation, we follow the
    # chain rule and compute the gradients as follows before accumulating
    # them all into a final grad_input.
    #  1) 'grad of out wrt var(X)' * 'grad of var(X) wrt X'
    #  2) 'grad of out wrt mean(X)' * 'grad of mean(X) wrt X'
    #  3) 'grad of out wrt X in the numerator' * 'grad of X wrt X'
    # We then rewrite the formulas to use as few extra buffers as possible
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps) ** 2  # d_denom = -num / denom**2
    # It is useful to delete tensors when you no longer need them with `del`
    # For example, we could've done `del tmp` here because we won't use it later
    # In this case, it's not a big difference because tmp only has size of (C,)
    # The important thing is avoid allocating NCHW-sized tensors unnecessarily
    d_var = d_denom / (2 * sqrt_var)  # denom = torch.sqrt(var) + eps
    # Compute d_mean_dx before allocating the final NCHW-sized grad_input buffer
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    # d_mean_dx has already been reassigned to a C-sized buffer so no need to worry

    # (1) unbiased_var(x) = ((X - unsqueeze_all(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    # (2) mean (see above)
    grad_input += d_mean_dx
    # (3) Add 'grad_out / <factor>' without allocating an extra buffer
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)  # sqrt_var + eps > 0!
    return grad_input


class FusedConvBN2DFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X, conv_weight, stride=1, padding=1, eps=1e-3):
        assert X.ndim == 4  # N, C, H, W
        # (1) Only need to save this single buffer for backward!
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(X, conv_weight)

        # (2) Exact same Conv2D forward from example above
        X = F.conv2d(X, conv_weight, stride=stride, padding=padding)
        # (3) Exact same BatchNorm2D forward from example above
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        # Try to do as many things in-place as possible
        # Instead of `out = (X - a) / b`, doing `out = X - a; out /= b`
        # avoids allocating one extra NCHW-sized buffer here
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_out):
        X, conv_weight = ctx.saved_tensors
        # (4) Batch norm backward
        # (5) We need to recompute conv
        X_conv_out = F.conv2d(X, conv_weight, stride=ctx.stride, padding=ctx.padding)
        grad_out = batch_norm_backward(grad_out, X_conv_out, ctx.sum, ctx.sqrt_var,
                                       ctx.N, ctx.eps)
        # (6) Conv2d backward
        grad_X, grad_input = convolution_backward(grad_out, X, conv_weight, ctx.stride, ctx.padding)
        return grad_X, grad_input, None, None, None, None, None


class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, exp_avg_factor=0.1,
                 eps=1e-3, device=None, dtype=None):
        super(FusedConvBN, self).__init__()
        assert stride == 1
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Conv parameters
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(*weight_shape, **factory_kwargs))
        # Batch norm parameters
        num_features = out_channels
        self.num_features = num_features
        self.eps = eps
        self.stride = stride
        self.padding = padding
        # Initialize
        self.reset_parameters(in_channels, kernel_size)

    def forward(self, X):
        return FusedConvBN2DFunction.apply(X, self.conv_weight, self.stride, self.padding, self.eps)

    def reset_parameters(self, in_channels, kernel_size) -> None:
        n = in_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_weight.data.uniform_(-stdv, stdv)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        if stride != 1:
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                FusedConvBN(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1)
            )
        else:
            self.residual_function = nn.Sequential(
                FusedConvBN(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                FusedConvBN(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1)
            )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1:
            self.residual_function = nn.Sequential(
                FusedConvBN(in_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                FusedConvBN(out_channels, out_channels * BottleNeck.expansion, kernel_size=1)
            )
        else:
            self.residual_function = nn.Sequential(
                FusedConvBN(in_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                FusedConvBN(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                FusedConvBN(out_channels, out_channels * BottleNeck.expansion, kernel_size=1)
            )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            FusedConvBN(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18(num_classes=10):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=10):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=10):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=10):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=10):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    weight = torch.rand(5, 3, 3, 3, requires_grad=True, dtype=torch.double)
    X = torch.rand(2, 3, 28, 28, requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(FusedConvBN2DFunction.apply, (X, weight, 1, 1))
