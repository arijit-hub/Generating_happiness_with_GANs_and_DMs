## CycleGAN using Pytorch Lightning ##

This codebase is highly inspired from Hadrien's awesome implementation of [CycleGAN](https://github.com/HReynaud/CycleGan_PL/tree/main) using Pytorch Lightning.

This implementation is based on the [Ultrasound Enhancement Challenge 2023](https://ultrasoundenhance2023.grand-challenge.org/registration/) dataset. To run this codebase on the corresponding dataset, download the data from the link, and save it in a folder (I prefer to keep it in `assets` folder). To perfectly align with the working, you can create a csv file with an added column of `split` which indicates the dataset splits for training, validation and testing. In case you want to use your own custom dataset please change the file `data.py` inside `data` directory to suit your needs.

To train the model on the Ultrasound Enhancement Challenge dataset use the following command.
```bash
python train.py
```

In case you want to run using a cluster machine please follow the `train_batch.sh` file. The file is specific to single node 8 GPU use-case, but you can change as you like.

I also have provided a few other Generator models apart from the Resnet Generator used in the main CycleGAN [paper](https://arxiv.org/pdf/1703.10593.pdf). These include the basic Unet and the Unet++ Generators. These can be found in the `models/networks.py` file. The code for both of these models were entirely taken from [this](https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py) amazing repository.

I achieved the highest score (rank 13) in the competition using a basic CycleGAN with Unet++ architecture. All the hyperparameters are kept intact in the `train.py` file for anyone to use and recreate. 

Hope this helps! :grinning:
