"""Implements pytorch lightning module for cyclegan."""
import sys

sys.path.append("utils")  # Run from the cyclegan_baseline folder

import torch
import torch.nn as nn
import itertools
import pytorch_lightning as pl
import lightning as L
from .loss import CycleGANLoss
from metrics import Metrics
from utility import make_img_grid


class CycleGAN(L.LightningModule):
    """Implements the entire Cyclegan module."""

    def __init__(
        self,
        source_2_target_generator: nn.Module,
        target_2_source_generator: nn.Module,
        source_discriminator: nn.Module,
        target_discriminator: nn.Module,
        gen_source_imgs_history,
        gen_target_imgs_history,
        generator_optim: torch.optim = torch.optim.Adam,
        discriminator_optim: torch.optim = torch.optim.Adam,
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        generator_adversarial_loss_factor: tuple = (1.0, 1.0),
        discriminator_adversarial_loss_factor: tuple = (0.5, 0.5),
        cycle_loss_factor: tuple = (1.0, 1.0),
        identity_loss_factor: tuple = (1.0, 1.0),
        lambda_generator_adversarial_loss: float = 1.0,
        lambda_discriminator_adversarial_loss: float = 1.0,
        lambda_identity_loss: float = 1.0,
        lambda_cycle_loss: float = 10.0,
    ):
        """Constructor.

        Parameters
        ----------
        source_2_target_generator: nn.Module
            The generator module for image translation
            from source domain to target domain.

        target_2_source_generator: nn.Module
            The generator module for image translation
            from target domain to source domain.

        source_discriminator: nn.Module
            The discriminator module for source domain image
            fake or real detection.

        target_discriminator: nn.Module
            The discriminator module for source domain image
            fake or real detection.

        gen_source_imgs_history : torch.tensor
            The container for storing a history of generated
            source domain images.

        gen_target_imgs_history : torch.tensor
            The container for storing a history of generated
            target domain images.

        generator_optim: torch.optim
            The optimizer class to train the generator modules.
            [Default : torch.optim.Adam]

        discriminator_optim: torch.optim
            The optimizer class to train the discriminator
            modules.
            [Default : torch.optim.Adam]

        generator_lr: float
            The respective learning rate for optimization of
            the generator modules.
            [Default : 0.0002]

        discriminator_lr: float
            The discriminator modules' learning rate for training.
            [Default : 0.0002]

        generator_adversarial_loss_factor: tuple
            The factor to multiply the individual generator
            adversarial loss value.
            [Default : (1.0,1.0)]

        discriminator_adversarial_loss_factor: tuple
            The factor to multiply the individual discriminator
            adversarial loss value.
            [Default : (0.5, 0.5)]

        cycle_loss_factor: tuple
            The factor to multiply the individual generator cycle
            loss value.
            [Default : (1.0,1.0)]

        identity_loss_factor: tuple
            The factor to multiply the individual generator identity
            loss value.
            [Default : (1.0,1.0)]

        lambda_generator_adversarial_loss : float
            The lambda factor to multiply the total generator
            adversarial loss.
            [Default : 1]

        lambda_discriminator_adversarial_loss : float
            The lambda factor to multiply the total discriminator
            adversarial loss.
            [Default : 1]

        lambda_identity_loss : float
            The lambda factor to multiply the identity loss.
            [Default : 1]

        lambda_cycle_loss : float
            The lambda factor to multiply the total cyclic loss.
            [Default : 10]
        """

        super().__init__()

        ## Loading the generator and discriminator instances ##
        self.source_2_target_generator = source_2_target_generator
        self.target_2_source_generator = target_2_source_generator
        self.source_discriminator = source_discriminator
        self.target_discriminator = target_discriminator

        ## Saving the generated images history containers ##
        self.gen_source_imgs_history = gen_source_imgs_history
        self.gen_target_imgs_history = gen_target_imgs_history

        ## Saving the optimizer classes ##
        self.generator_optim = generator_optim
        self.discriminator_optim = discriminator_optim

        ## Saving the learning rates ##
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        ## Setting to manual optimization ##
        self.automatic_optimization = False

        ## Instantiating the loss function ##
        self.loss = CycleGANLoss()

        ## Saving the loss factors ##
        self.generator_adversarial_loss_factor = generator_adversarial_loss_factor
        self.discriminator_adversarial_loss_factor = (
            discriminator_adversarial_loss_factor
        )
        self.cycle_loss_factor = cycle_loss_factor
        self.identity_loss_factor = identity_loss_factor

        ## Saving the lambda factors ##
        self.lambda_generator_adversarial_loss = lambda_generator_adversarial_loss
        self.lambda_discriminator_adversarial_loss = (
            lambda_discriminator_adversarial_loss
        )
        self.lambda_identity_loss = lambda_identity_loss
        self.lambda_cycle_loss = lambda_cycle_loss

        ## Saving the metrics container ##
        self.metrics = Metrics()

        ## Save hyperparameters ##
        self.save_hyperparameters()

    def _get_parameters(self):
        """Returns the generator and the discriminator parameters.
        The two generator parameters are chained together and the
        two discriminator parameters are chained together and
        are sent out separately.

        Returns
        -------
        list
            Two groups of parameters.
        """

        generator_params = itertools.chain(
            self.source_2_target_generator.parameters(),
            self.target_2_source_generator.parameters(),
        )

        discriminator_params = itertools.chain(
            self.source_discriminator.parameters(),
            self.target_discriminator.parameters(),
        )

        return generator_params, discriminator_params

    def forward(self, source_img, target_img=None):
        """The generic forward pass.

        Parameters
        ----------
        source_img : torch.tensor
            The source domain image.

        target_img : torch.tensor
            The target domain images.

        Returns
        -------
        tuple
            A collection of the generated target image,
            generated source image and the reconstructed
            target and source image.
        """
        if target_img == None:
            gen_target_img = self.source_2_target_generator(source_img)
            return gen_target_img

        else:
            gen_target_img = self.source_2_target_generator(source_img)
            gen_source_img = self.target_2_source_generator(target_img)

            recon_target_img = self.source_2_target_generator(gen_source_img)
            recon_source_img = self.target_2_source_generator(gen_target_img)

            return gen_target_img, gen_source_img, recon_target_img, recon_source_img

    def configure_optimizers(self):
        """Configures optimizers for the generator and discriminator
        models.
        """
        generator_params, discriminator_params = self._get_parameters()

        ## Setting the optimizers of the generator and discriminator ##
        if self.generator_optim == torch.optim.Adam:
            generator_optim = self.generator_optim(
                generator_params, self.generator_lr, betas=(0.5, 0.999)
            )
        else:
            generator_optim = self.generator_optim(generator_params, self.generator_lr)

        if self.discriminator_optim == torch.optim.Adam:
            discriminator_optim = self.discriminator_optim(
                discriminator_params, self.discriminator_lr, betas=(0.5, 0.999)
            )
        else:
            discriminator_optim = self.discriminator_optim(
                discriminator_params, self.discriminator_lr
            )
        ## Setting learning rate multiplicative factor ##
        lambda_lr = (
            # lambda epoch: 1
            lambda epoch: max(
                0, (self.trainer.max_epochs - epoch) / (self.trainer.max_epochs // 2)
            )
            if epoch > (self.trainer.max_epochs // 2)
            else 1
        )

        ## Setting the schedulers ##
        generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            generator_optim, lambda_lr
        )
        discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            discriminator_optim, lambda_lr
        )

        return [generator_optim, discriminator_optim], [
            generator_scheduler,
            discriminator_scheduler,
        ]

    def _shared_step(self, batch, batch_idx, split="train"):
        """This resembles the similar working mechanism step
        for the training and the validation step with a bit
        of difference.

        Parameters
        ----------
        batch : tuple
        A minibatch of tuple of source and target domain
            images.

        batch_idx : int
            The current minibatch index.

        split : str
            The training or the validation split.
            [Default : "train"]
            [Options : "train" or "val"]
        """

        source_imgs, target_imgs = batch

        batch_size = source_imgs.shape[0]

        gen_target_img, gen_source_img, recon_target_img, recon_source_img = self(
            source_imgs, target_imgs
        )

        ## Fetching the optimizers ##
        if split == "train":
            generator_optim, discriminator_optim = self.optimizers()

        ## Generator optimization ##
        ## Generator loss consists of the adversarial loss, the cycle loss...#
        # and the identity loss ##

        ## First, fetching the adversarial loss for both the generator ##
        source_2_target_generator_adversarial_loss = (
            self.loss.calculate_single_generator_adversarial_loss(
                generated_end_domain_imgs=gen_target_img,
                end_domain_discriminator=self.target_discriminator,
                factor=self.generator_adversarial_loss_factor[0],
            )
        )

        target_2_source_generator_adversarial_loss = (
            self.loss.calculate_single_generator_adversarial_loss(
                generated_end_domain_imgs=gen_source_img,
                end_domain_discriminator=self.source_discriminator,
                factor=self.generator_adversarial_loss_factor[1],
            )
        )
        generator_adversarial_loss = (
            source_2_target_generator_adversarial_loss
            + target_2_source_generator_adversarial_loss
        )

        ## Cycle consistency loss for both the generator ##
        cycle_loss = self.loss.calculate_cycle_consistency_loss(
            start_domain_imgs=source_imgs,
            end_domain_imgs=target_imgs,
            recon_start_domain_imgs=recon_source_img,
            recon_end_domain_imgs=recon_target_img,
            factor=self.cycle_loss_factor,
        )

        ## Identity loss of the generator ##

        if self.lambda_identity_loss != 0:
            identity_loss = self.loss.calculate_identity_loss(
                start_2_end_domain_generator=self.source_2_target_generator,
                end_2_start_domain_generator=self.target_2_source_generator,
                start_domain_imgs=source_imgs,
                end_domain_imgs=target_imgs,
                factor=self.identity_loss_factor,
            )

        else:
            identity_loss = torch.zeros_like(cycle_loss)

        ## Finally calculating the total generator loss ##
        generator_loss = (
            self.lambda_generator_adversarial_loss * generator_adversarial_loss
            + self.lambda_cycle_loss * cycle_loss
            + self.lambda_identity_loss * identity_loss
        )

        ## Optimizing the generator ##
        if split == "train":
            self.toggle_optimizer(generator_optim)
            generator_optim.zero_grad()
            self.manual_backward(generator_loss)
            generator_optim.step()
            self.untoggle_optimizer(generator_optim)

        ## Discriminator optimization ##
        ## Just return the discriminator adversarial loss ##

        source_disc_loss = self.loss.calculate_single_discriminator_adversarial_loss(
            gen_ending_domain_history=self.gen_source_imgs_history,
            generated_ending_domain_imgs=gen_source_img,
            ending_domain_imgs=source_imgs,
            end_domain_discriminator=self.source_discriminator,
            factor=self.discriminator_adversarial_loss_factor[0],
        )

        target_disc_loss = self.loss.calculate_single_discriminator_adversarial_loss(
            gen_ending_domain_history=self.gen_target_imgs_history,
            generated_ending_domain_imgs=gen_target_img,
            ending_domain_imgs=target_imgs,
            end_domain_discriminator=self.target_discriminator,
            factor=self.discriminator_adversarial_loss_factor[1],
        )

        ## Finally calculating the total discriminator loss ##
        discriminator_loss = self.lambda_discriminator_adversarial_loss * (
            source_disc_loss + target_disc_loss
        )

        ## Optimization discriminator ##
        if split == "train":
            self.toggle_optimizer(discriminator_optim)
            discriminator_optim.zero_grad()
            self.manual_backward(discriminator_loss)
            discriminator_optim.step()
            self.untoggle_optimizer(discriminator_optim)

        ## Getting metrics ##
        ssim_target, psnr_target, lncc_target = self.metrics.get_metrics(
            gen_target_img, target_imgs
        )
        ssim_source, psnr_source, lncc_source = self.metrics.get_metrics(
            gen_source_img, source_imgs
        )

        ## Making the llogging dict ##

        log_output = {
            split + "/total_loss": (generator_loss + discriminator_loss).detach(),
            split + "/total_gen_loss": generator_loss.detach(),
            split + "/total_disc_loss": discriminator_loss.detach(),
            split
            + "/source_2_target_gen_adv_loss": source_2_target_generator_adversarial_loss.detach(),
            split
            + "/target_2_source_gen_adv_loss": target_2_source_generator_adversarial_loss.detach(),
            split + "/identity_loss": identity_loss.detach(),
            split + "/cycle_loss": cycle_loss.detach(),
            split + "/source_disc_loss": source_disc_loss.detach(),
            split + "/target_disc_loss": target_disc_loss.detach(),
            split + "/ssim_target_domain": ssim_target.detach(),
            split + "/ssim_source_domain": ssim_source.detach(),
            split + "/psnr_target_domain": psnr_target.detach(),
            split + "/psnr_source_domain": psnr_source.detach(),
            split + "/lncc_target_domain": lncc_target.detach(),
            split + "/lncc_target_source": lncc_source.detach(),
        }

        self.log_dict(
            log_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        if split == "train":
            total_batches = self.trainer.num_training_batches

        else:
            total_batches = self.trainer.num_val_batches [0]

        ## Logging imgaes ##
        if batch_idx == (total_batches - 1) and self.local_rank == 0:
            make_img_grid(imgs = [source_imgs, gen_target_img, target_imgs],
                                  imgs_labels=[f'{split} source', f'{split} pred', f'{split} target'],
                                  normalize=True,
                                  output_dir='generated_imgs',
                                  output_file_name=f'{self.current_epoch}_{split}_img.png',)

    def training_step(self, batch, batch_idx):
        """Generic Training step of a pytorch lightning module.

        Parameters
        ----------
        batch : torch.data.utils.DataLoader
            A minibatch of tuple of source and target domain
            images.

        batch_idx : int
            The current minibatch index.
        """

        return self._shared_step(batch, batch_idx, split="train")

    def validation_step(self, batch, batch_idx):
        """Generic Validation step of a pytorch lightning module.

        Parameters
        ----------
        batch : torch.data.utils.DataLoader
            A minibatch of tuple of source and target domain
            images.

        batch_idx : int
            The current minibatch index.
        """

        return self._shared_step(batch, batch_idx, split="val")

    def on_train_epoch_end(self):
        """Generic pytorch lightning utility to do
        the schedule update after each epoch.
        """

        generator_scheduler, discriminator_scheduler = self.lr_schedulers()

        generator_scheduler.step()
        discriminator_scheduler.step()
