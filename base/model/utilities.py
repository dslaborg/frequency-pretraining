from math import ceil


def pad_for_same(
    input_size: int, kernel: int = 1, stride: int = 1, dilation: int = 1
) -> tuple[int, int]:
    """
    Calculate padding for 'same' padding in convolutional layers. This is the padding that makes the output of the
    convolutional layer the same size as the input.

    :param input_size: size of the input
    :param kernel: size of the kernel
    :param stride: stride of the convolution
    :param dilation: dilation of the convolution

    :return: tuple of padding for the left and right side
    """
    pad_total = max(
        (ceil(input_size / stride) - 1) * stride
        + (kernel - 1) * dilation
        + 1
        - input_size,
        0,
    )
    return int(pad_total // 2), int(pad_total - pad_total // 2)
