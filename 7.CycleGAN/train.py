import torch
from data.data import EnhanceDataModule
from models.model import CycleGAN
from models.networks import Generator, Discriminator, UNet, NestedUNet
# from utils.image_history import ImageHistory
from utils.image_pool import ImagePool
from utils.utility import init_weights
from torchvision.transforms import transforms
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar


torch.set_float32_matmul_precision("high")

seed_everything(97, workers=True)

normalization_trans = lambda x: (x - 0.5) / 0.5

train_transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        # transforms.RandomRotation(degrees=30),
        # transforms.RandomHorizontalFlip(p=0.4),
        # transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Lambda(normalization_trans),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Lambda(normalization_trans),
    ]
)

## Setting the dataloading ##
enhance_data = EnhanceDataModule(
    csv_path = 'assets/images.csv',
    root_img_path='assets',
    img_type="all",
    batch_size=1
)

## Setting the models ##
## 1. Source -> Target domain generator ##
source_2_target_generator = Generator(
    depth=3,
    num_res_blocks=9,
    in_channels=1,
    out_channels=1,
    initial_num_filters=64,
    norm="batch",
    act="relu",
)
# source_2_target_generator = NestedUNet(
#     input_channels=3,
#     output_channels=3
# )

source_2_target_generator.apply(init_weights)

## 2. Target -> Source domain generator ##
target_2_source_generator = Generator(
    depth=3,
    num_res_blocks=9,
    in_channels=1,
    out_channels=1,
    initial_num_filters=64,
    norm="batch",
    act="relu",
)

# target_2_source_generator = NestedUNet(
#     input_channels=3,
#     output_channels=3
# )

target_2_source_generator.apply(init_weights)

## 3. Source domain real/fake discriminator ##
source_discriminator = Discriminator(
    in_channels=1,
    initial_num_filters=64,
    num_blocks=3,
    norm="instance",
    act="leaky",
    kernel_size=4,
    stride=2,
)

source_discriminator.apply(init_weights)

## 4. Target domain real/fake discriminator ##
target_discriminator = Discriminator(
    in_channels=1,
    initial_num_filters=64,
    num_blocks=3,
    norm="instance",
    act="leaky",
    kernel_size=4,
    stride=2,
)

target_discriminator.apply(init_weights)

## Setting the CycleGAN model ##
model = CycleGAN(
    source_2_target_generator=source_2_target_generator,
    target_2_source_generator=target_2_source_generator,
    source_discriminator=source_discriminator,
    target_discriminator=target_discriminator,
    generator_lr=0.0001,
    discriminator_lr=0.0001,
    discriminator_adversarial_loss_factor= (1., 1.),
    gen_source_imgs_history=ImagePool(pool_size=50),  
    gen_target_imgs_history=ImagePool(pool_size=50),  
    lambda_identity_loss=0.,
)

## Setting the logger ##
tb_logger = pl_loggers.TensorBoardLogger(save_dir="results/")

## Setting the model checkpoint save callback to save model ...#
## on getting lower validation loss ##
checkpoint_callback = ModelCheckpoint(
    dirpath="assets/checkpoints/",
    filename="best_model",
    verbose=True,
    monitor="val/total_gen_loss",
    mode="min",
)

## Finally setting the trainer ##
trainer = Trainer(
    logger=tb_logger,
    max_epochs=200,
    callbacks=[RichProgressBar(leave=True), checkpoint_callback],
    inference_mode=True,
    accelerator="gpu",
    devices=8,
    enable_model_summary=True,
    strategy="ddp_find_unused_parameters_true",
    num_nodes=1,
    num_sanity_val_steps=0,  # Skip Sanity Check
)

trainer.fit(model, enhance_data)
