"""Implements the evaluation metrics for the CycleGAN model."""

import torch
from torchmetrics.functional import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
)
from monai.losses import LocalNormalizedCrossCorrelationLoss


class Metrics:
    """Container for all the metrics used for evaluation."""

    def __init__(self):
        """Constructor"""

        ## Instanting a LocalNormalizedCrossCorrelationLoss object ##
        self.lncc = LocalNormalizedCrossCorrelationLoss(spatial_dims=2)

    def get_metrics(self, gen_imgs, real_imgs):
        """Calculates and Returns the metrics used for
        evaluation.

        Parameters
        ----------
        gen_imgs : torch.tensor
            The generated images.

        real_imgs : torch.tensor
            The real images.
        """

        ## Structural Similarity Index Measure ##
        ssim = structural_similarity_index_measure(gen_imgs, real_imgs)

        ## Peak Signal to Noise Ratio ##
        psnr = peak_signal_noise_ratio(gen_imgs, real_imgs)

        ## Local Normalized Cross Correlation ##
        lncc = -1 * self.lncc(gen_imgs, real_imgs)

        return ssim, psnr, lncc
