import torch
from dataset.customDataset import CustomDataSet
from util.utility import init
from hyperparameters import infer_conf, INFER_EX_DIR
from model.customModel2d import CustomModel
from config.configClass import Config
from ckptmanager.manager import CheckPointManager
from torch.utils.data import DataLoader
from valider.valider import Valider
from loss.customLoss import CustomLoss
from evaluator.Loss2evaluator import LossScore
from torchmetrics import MetricCollection
import matplotlib.pyplot as plt
import numpy as np
from util.io import clear_folder

clear_folder(INFER_EX_DIR)

def main(conf:Config) -> None:
    """Inferencing model base on given config."""

    # device setting
    device = torch.device(conf.device)

    # setup model and other components
    model = CustomModel(conf)                                                # create model

    # load model from checkpoint if needed
    ckpt_manager = CheckPointManager(conf, model)
    if conf.Load:
        ckpt_manager.load(ckpt_path=conf.load_path, device=device)

    # prepare test dataset and dataloader
    test_dataset = CustomDataSet(conf, conf.test_dataset)       # create test dataset
    batch_size = conf.batch_size
    num_workers = conf.num_workers
    loader = DataLoader(dataset     = test_dataset,
                        batch_size  = batch_size,
                        shuffle     = True,
                        pin_memory  = True,
                        num_workers = num_workers)  # create train dataloader

    criterion = CustomLoss()
    eval = MetricCollection({'val_loss':LossScore(criterion)})

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader, start=1):
            if batch_idx == 20:
                break
            # move input and label to device
            input = input.to(device)
            label = label.to(device)
            # forward
            output:torch.Tensor = model(input)
            from model.CCNN import CLayerNorm
            layer_norm = CLayerNorm([80, 80]).to(device)
            input = layer_norm(input).squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            input_intensity = input[...,0]**2 + input[...,1]**2
            output = layer_norm(output).squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            output_intensity = output[...,0]**2 + output[...,1]**2
            label = layer_norm(label).squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            label_intensity = label[...,0]**2 + label[...,1]**2
            # save output as image
            plt.figure()
            plt.imshow(input_intensity)
            plt.savefig(f'{INFER_EX_DIR}/{batch_idx}_input.png')
            plt.close()

            plt.figure()
            plt.imshow(output_intensity)
            plt.savefig(f'{INFER_EX_DIR}/{batch_idx}_output.png')
            plt.close()

            plt.figure()
            plt.imshow(label_intensity)
            plt.savefig(f'{INFER_EX_DIR}/{batch_idx}_label.png')
            plt.close()


if __name__ == '__main__':
    #%% print version information
    print(f'Torch version: {torch.__version__}')
    # initialize
    init(infer_conf.seed)

    #%% run main function
    main(infer_conf)
