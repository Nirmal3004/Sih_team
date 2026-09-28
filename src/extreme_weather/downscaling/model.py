"""The learned downscaling network (SIH26078, Phase 4).

> **Research prototype trained on synthetic data.** Its output is not a forecast
> and carries no guarantee of real-world accuracy.

Architecture
------------
```text
input  [B, C_in, H, W]          C_in = 10 documented feature channels
   |
   v
feature projection                 1x1 conv -> C hidden channels, ReLU
   |
   v
residual block  x N                3x3 conv -> ReLU -> 3x3 conv, + skip
   |
   v
output conv                       1x1 conv -> 1 channel  (the RESIDUAL)
   |
   v
ReLU                              non-negativity by construction
   |
   v
upsample to the target grid        bilinear interpolation to (H_t, W_t)
   |
   v
add the baseline                   AI_output = baseline + predicted_residual
```

Residual learning
-----------------
The network predicts a **residual on top of a bilinear baseline**, not the field
itself:

```text
ai_output = bilinear(coarse) + ReLU(residual_net(features))
```

Two reasons. First, the bilinear upsample already satisfies the coarse-scale
constraint (its area average reproduces the coarse cell), so the network only has
to learn the *sub-cell* correction - a small, zero-centred quantity that is far
easier to fit than the full field. Second, it bounds the damage: a network that
learns nothing returns the baseline exactly, rather than something worse.

The residual passes through **ReLU**, so the correction is non-negative by
construction and the output cannot be pulled below the baseline. That is a
deliberate, documented constraint, not a silent clip: it is stated here, stored
in the checkpoint and asserted by tests. The trade-off is that the network cannot
learn to *suppress* the baseline where the true field is lower; a sigmoid-gated
variant would allow that and is listed as future work.

Why a small CNN
---------------
The task is spatially local (neighbourhoods of a few coarse cells), which is
exactly what a convolution suits, and the dataset is small. The network trains in
seconds on CPU. PyTorch is already a project dependency; **PyTorch Lightning,
PyTorch Geometric, diffusion and GAN frameworks are deliberately not used**, and
no dependency is added in this phase.

Flexibility
-----------
The scale factor is read from the data, never assumed. The network runs on the
coarse grid at its native size and the upsample happens in the forward pass, so
any target grid is handled without retraining.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from extreme_weather.downscaling.baseline import DownscalingError
from extreme_weather.downscaling.features import FEATURE_COUNT

__all__ = [
    "MODEL_VERSION",
    "DownscalingError",
    "DownscalingModel",
    "ModelConfig",
    "ResidualBlock",
    "init_generator",
]

#: Bumped when the parameter layout changes, so an old checkpoint cannot be
#: silently loaded into a different architecture.
MODEL_VERSION = "downscaling-cnn-1.0"


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of the downscaling network.

    Attributes:
        in_channels: Number of input feature channels (10).
        hidden_channels: Width of the internal representation.
        num_blocks: Number of residual blocks.
        output_activation: ``"relu"`` enforces non-negativity of the residual.
            It is the only supported value in this phase.
    """

    in_channels: int = FEATURE_COUNT
    hidden_channels: int = 32
    num_blocks: int = 3
    output_activation: str = "relu"

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("in_channels and hidden_channels must be > 0.")
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}.")
        if self.output_activation != "relu":
            raise ValueError(
                f"Only 'relu' is supported in this phase (it guarantees a non-negative "
                f"residual); got {self.output_activation!r}."
            )

    def as_dict(self) -> dict[str, int | str]:
        """Plain-dict view, stored in the checkpoint."""
        return dict(asdict(self))


def init_generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded with ``seed``.

    An explicit generator (not the global RNG) makes initialisation reproducible
    regardless of what else ran before it.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x + conv2(relu(conv1(x)))``."""
        return x + self.conv2(self.activation(self.conv1(x)))


class DownscalingModel(nn.Module):
    """Predict a non-negative residual that is added to a bilinear baseline.

    Args:
        config: Architecture hyper-parameters.

    Shape:
        - input: ``[B, in_channels, H, W]`` at the **coarse** grid size.
        - output: ``[B, 1, H_target, W_target]`` at the requested fine size.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.projection = nn.Conv2d(cfg.in_channels, cfg.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualBlock(cfg.hidden_channels) for _ in range(cfg.num_blocks)
        )
        self.output_conv = nn.Conv2d(cfg.hidden_channels, 1, kernel_size=1)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise every weight from the seeded generator."""
        generator = init_generator(20260601)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                bound = 1.0 / math.sqrt(module.in_channels * module.kernel_size[0] ** 2)
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound, generator=generator)
                    if module.bias is not None:
                        module.bias.zero_()

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        features: Tensor,
        target_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        """Predict the non-negative residual at the requested fine size.

        Args:
            features: ``[B, in_channels, H, W]`` coarse-grid features.
            target_shape: Requested ``(H_target, W_target)``. ``None`` keeps the
                input size (no spatial change).

        Returns:
            ``[B, 1, H_target, W_target]`` non-negative residual, in the
            variable's physical units (mm h-1).

        Raises:
            DownscalingError: If the channel count, rank or target shape is
                invalid.
        """
        if features.ndim != 4:
            raise DownscalingError(
                f"features must be [B, C, H, W]; got {tuple(features.shape)}."
            )
        if features.shape[1] != self.config.in_channels:
            raise DownscalingError(
                f"Expected {self.config.in_channels} input channels, got {features.shape[1]}."
            )
        if features.shape[0] < 1 or features.shape[2] < 1 or features.shape[3] < 1:
            raise DownscalingError(f"Empty input tensor {tuple(features.shape)}.")

        if target_shape is None:
            target_shape = (int(features.shape[2]), int(features.shape[3]))
        if len(target_shape) != 2 or target_shape[0] < 1 or target_shape[1] < 1:
            raise DownscalingError(f"Invalid target shape {target_shape}.")

        hidden = self.activation(self.projection(features))
        for block in self.blocks:
            hidden = block(hidden)
        # ReLU on the output is the documented non-negativity constraint: the
        # network may only add structure to the baseline, never subtract from it.
        residual = self.activation(self.output_conv(hidden))

        # Upsample the residual to the requested fine grid; bilinear keeps the
        # correction smooth, since the fine structure is learned upstream.
        if tuple(residual.shape[-2:]) != tuple(target_shape):
            residual = F.interpolate(
                residual, size=tuple(target_shape), mode="bilinear", align_corners=False
            )
        return residual

