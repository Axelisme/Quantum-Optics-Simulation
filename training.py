import time
import wandb
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from util.utility import init
from hyperparameters import train_conf
from model.customModel2d import SixLayerModel as CustomModel
from loss.customLoss import CustomLoss
from evaluator.Loss2evaluator import LossScore
from valider.valider import Valider
from dataset.labelDataset import CustomDataSet
from trainer.trainer import Trainer
from config.configClass import Config
from ckptmanager.manager import CheckPointManager


def start_train(conf:Config):
    """Training model base on given config."""

    # device setting
    device = torch.device(conf.device)

    # setup model and other components
    model = CustomModel(conf)                                                               # create model
    optimizer = AdamW(model.parameters(), lr=conf.init_lr)                                  # create optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) # create scheduler
    criterion = CustomLoss()                                                         # create criterion
    eval1 = LossScore(criterion)
    evaluators = MetricCollection({'val_loss':eval1})

    # load model and optimizer from checkpoint if needed
    ckpt_manager = CheckPointManager(conf, model, optim=optimizer, scheduler=scheduler, lower_better=conf.lower_better)
    ckpt_manager.save_config(f"train_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
    if conf.Load:
        ckpt_manager.load(ckpt_path=conf.load_path, device=device)

    # register model to wandb if needed
    if conf.WandB and not hasattr(conf,"Sweep"):
        wandb.watch(models=model, criterion=criterion, log="gradients", log_freq=100)

    # prepare dataset and dataloader
    train_transform = None
    valid_transform = None
    train_set = CustomDataSet(conf, conf.train_dataset, transform=train_transform)    # create train dataset
    valid_set = CustomDataSet(conf, conf.valid_dataset, transform=valid_transform)    # create valid dataset
    batch_size = conf.batch_size
    num_workers = conf.num_workers
    train_loader = DataLoader(dataset     = train_set,
                              batch_size  = batch_size,
                              shuffle     = True,
                              pin_memory  = True,
                              num_workers = num_workers)  # create train dataloader
    valid_loader = DataLoader(dataset     = valid_set,
                              batch_size  = batch_size,
                              shuffle     = False,
                              pin_memory  = True,
                              num_workers = num_workers)  # create valid dataloader

    # create trainer and valider
    trainer = Trainer(conf, model, device, train_loader, optimizer, criterion)
    valider = Valider(conf, model, device, valid_loader, evaluators)

    # start training
    save_metric = conf.save_metric
    for epoch in range(1,conf.epochs+1):
        print('-'*79)

        train_result = trainer.fit()                                                # train a epoch
        valid_result = valider.eval()                                               # validate a epoch

        lr = optimizer.param_groups[0]['lr']                                        # get current learning rate
        show_result(conf, epoch, lr, train_result, valid_result)                    # show result

        cur_score = valid_result[save_metric].item()                            # get current score
        scheduler.step(metrics=cur_score)                                       # update learning rate

        if conf.Save:                                                               # save checkpoint if needed
            ckpt_manager.update(cur_score, epoch)                                   # save checkpoint if better

        if hasattr(conf,"WandB") and conf.WandB:                                    # log result to wandb
            wandb.log({'lr':lr}, step=epoch, commit=False)
            wandb.log({'train_loss':train_result}, step=epoch, commit=False)
            wandb.log(valid_result, step=epoch, commit=True)


def show_result(conf:Config, epoch, lr, train_result, valid_result:dict) -> None:
    """Print result of training and validation."""
    # print result
    print(f'Epoch: ({epoch} / {conf.epochs})')
    print(f'lr: {lr:0.3e}')
    print("Train result:")
    print(f'\ttrain_loss: {train_result:0.4f}')
    print("Valid result:")
    for name, evaluator in valid_result.items():
        print(f'\t{name}: {evaluator:0.4f}')


if __name__ == '__main__':
    # print information
    print(f'Torch version: {torch.__version__}')
    # initialize
    init(train_conf.seed)
    if hasattr(train_conf,"WandB") and train_conf.WandB:
        wandb.init(project=train_conf.project_name,
                   name=train_conf.model_name,
                   config=train_conf.as_dict())

    # start training
    start_train(train_conf)
