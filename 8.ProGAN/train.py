"""Implements the training for ProGAN.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data import CelebADataset
from networks import ProGANGenerator, ProGANDiscriminator
from loss import generator_loss, discriminator_loss

from torchvision.utils import save_image

from torchvision.transforms.functional import resize

from tqdm import tqdm


def train(num_epochs):
    """Sets up the training loop.

    Parameters
    ----------
    num_epochs : int
        The number of epochs to train the model for.
    """

    ## Setting the device ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Setting the hyperparameters ##
    img_channels = 3
    batch_size = 4
    learning_rate = 0.001
    num_blocks = 7
    num_initial_filters = 16
    latent_dim = 512  # This is the noise dimension
    num_workers = 8
    dataset_root_dir = "img_align_celeba/img_align_celeba"
    log_dir = "results"
    c = 10
    current_num_block = 1
    last_num_block = None
    max_num_block_reached_flag = False

    ## Setting up a fixed noise batch for visualization ##
    fixed_noise_batch = torch.randn(batch_size, latent_dim, 1, 1).to(device)

    ## Setting up the dataset ##
    dataset = CelebADataset(root_dir=dataset_root_dir)
    print(f"Dataset loaded wiht {len(dataset)} images")

    ## Setting up the dataloader ##
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    ## Setting up the generator ##
    generator = ProGANGenerator(
        num_blocks=num_blocks,
        latent_dim=latent_dim,
        num_initial_filters=num_initial_filters,
        out_channels=img_channels,
    ).to(device)

    ## Setting up the discriminator ##
    discriminator = ProGANDiscriminator(
        num_blocks=num_blocks,
        latent_dim=latent_dim,
        num_initial_filters=num_initial_filters,
        in_channels=img_channels,
    ).to(device)

    ## Setting up the optimizers ##
    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(), lr=learning_rate, betas=(0, 0.99)
    )

    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(), lr=learning_rate, betas=(0, 0.99)
    )

    ## Setting up the training ##

    for epoch in range(num_epochs):
        print(
            f"For debugging : Epoch = {epoch} -> Current Block = {current_num_block}, Last Block = {last_num_block}, Max Block Flag = {max_num_block_reached_flag}"
        )
        loop = tqdm(dataloader)

        epoch_gen_loss = []
        epoch_disc_loss = []

        alphas = torch.linspace(0, 1, len(loop))

        for idx, (real_images, _) in enumerate(loop):
            ## Preparing the real images ##
            real_images = real_images.to(device)

            ## Training the generator ##
            generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)

            if epoch % 4 == 0 and not max_num_block_reached_flag:
                alpha = alphas[idx]
            else:
                alpha = 1.0

            fake_images = generator(noise, current_num_block, alpha)

            ## Resizing the real images to match the fake ones ##
            real_images = resize(
                real_images, size=(fake_images.shape[2], fake_images.shape[3])
            )
            generator_loss_value = generator_loss(
                discriminator, current_num_block, alpha, fake_images
            )
            generator_loss_value.backward()
            generator_optimizer.step()

            epoch_gen_loss.append(generator_loss_value.item())

            ## Training the discriminator ##
            discriminator_optimizer.zero_grad()
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_images = generator(noise, current_num_block, alpha).detach()
            discriminator_loss_value = discriminator_loss(
                discriminator,
                current_num_block,
                alpha,
                real_images,
                fake_images,
                c=c,
            )
            discriminator_loss_value.backward()
            discriminator_optimizer.step()

            epoch_disc_loss.append(discriminator_loss_value.item())

            ## Updating the progress bar ##
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(
                gen_loss=generator_loss_value.item(),
                disc_loss=discriminator_loss_value.item(),
            )

        ## Saving the images ##
        with torch.no_grad():
            fake_images = generator(fixed_noise_batch, current_num_block, 1.0)
            fake_images = fake_images * 0.5 + 0.5
            save_image(
                resize(fake_images, (256, 256)),
                f"{log_dir}/images/fake_images-{epoch+1}.png",
                normalize=True,
                nrow=4,
            )

        ## Saving the model ##
        torch.save(
            generator.state_dict(),
            f"{log_dir}/checkpoints/generator.pth",
        )

        torch.save(
            discriminator.state_dict(),
            f"{log_dir}/checkpoints/discriminator.pth",
        )

        if last_num_block == (num_blocks - 1):
            max_num_block_reached_flag = True

        ## Updating the current number of blocks ##
        if current_num_block < num_blocks and (epoch + 1) % 4 == 0:
            last_num_block = current_num_block
            current_num_block += 1


if __name__ == "__main__":
    train(num_epochs=40)
