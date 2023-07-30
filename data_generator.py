
"""A script to generate data for the project."""
#%%
import os
import h5py
import numpy as np
from typing import Dict, List
from torch.utils.data import random_split
from util.io import clear_folder
from hyperparameters import base_conf, PROC_DATA_DIR
from data_qinn import get_wave_pair
import random
import multiprocessing as mp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# some parameters
dataset_name = "debugset"
split_ratio = base_conf.split_ratio
SAVE_DIR = os.path.join(PROC_DATA_DIR, dataset_name)
clear_folder(SAVE_DIR) # clear the folder before generating data
seed = 0
random.seed(seed)

#%%
def loader(_):
    Input,Ouput = get_wave_pair()
    Input = Input.reshape(1,80,80,80,2)
    Ouput = Ouput[-1,:,:,:,:]
    Ouput = Ouput.reshape(1,80,80,80,2)
    return Input,Ouput

def generate_process_data(save_dir:str, dataset_name:str, batch_max = -1) -> None:
    """
    Generate whole processed data for the project.
    """
    # create folder for processed data
    ALL_DATASET_PATH = os.path.join(save_dir, f"{dataset_name}_all.h5")

    # load data
    data_dtype = np.dtype([("input", np.float32, (1,80,80,80,2)), ("output", np.float32,(1,80,80,80,2))])
    data_loader = loader
    data_length = 1000

    if batch_max == -1:
        batch_max = data_length

    # save data to h5 file
    print(f"Writting total dataset to {ALL_DATASET_PATH}")
    with h5py.File(ALL_DATASET_PATH, mode='w') as writer:
        # write meta data
        writer.attrs["mode"] = "all"
        writer.attrs["length"] = data_length
        writer.attrs["ratio"] = 1.0
        writer.attrs["data_dtype"] = str(data_dtype)
        # write dataset
        dataset = writer.create_dataset("dataset", (data_length,), dtype=data_dtype)
        for batch_idx in tqdm(range(data_length//batch_max + 1), desc="Generating data"):
            batch_size = min(batch_max, data_length-batch_idx*batch_max)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                datas = pool.imap_unordered(data_loader, range(batch_size))
                for idx, data in enumerate(datas):
                    dataset[batch_idx*batch_max+idx] = data

generate_process_data(SAVE_DIR, dataset_name, batch_max=100)


#%%
def split_process_data(save_dir:str, dataset_name:str, split_ratio:Dict[str,float]) -> None:
    """
    Split the whole processed data into some dataset.
    """
    ALL_DATASET_PATH = os.path.join(save_dir, f"{dataset_name}_all.h5")

    with h5py.File(ALL_DATASET_PATH, mode='r') as reader:
        # load data
        dataset = reader["dataset"]
        # split dataset
        splited_datasets = random_split(dataset, list(split_ratio.values())) # type: ignore

        # save splited dataset
        for (mode, ratio), named_dataset in zip(split_ratio.items(), splited_datasets):
            NAMED_DATASET_PATH = os.path.join(save_dir, f"{dataset_name}_{mode}.h5")
            print(f"Writting {mode} dataset to {NAMED_DATASET_PATH}")

            # save data to h5 file
            with h5py.File(NAMED_DATASET_PATH, mode='w') as writer:
                # write meta data
                writer.attrs["mode"] = mode
                writer.attrs["length"] = len(named_dataset)
                writer.attrs["ratio"] = ratio
                writer.attrs["data_dtype"] = reader.attrs["data_dtype"]
                print(reader.attrs["data_dtype"])
                # write dataset
                dataset = writer.create_dataset("dataset", data=named_dataset)

split_process_data(SAVE_DIR, dataset_name, split_ratio)


#%%
def sampling_process_samples(save_dir, dataset_name:str, modes:List[str], num = 100):
    """
    Sampling some samples from the processed data.
    """
    for mode in modes:
        DATASER_PATH = os.path.join(save_dir, f"{dataset_name}_{mode}.h5")
        SAMPLES_DIR = os.path.join(save_dir, f"{mode}_samples")
        clear_folder(SAMPLES_DIR)
        with h5py.File(DATASER_PATH, mode='r') as reader:
            # load data
            dataset = reader["dataset"]
            # sampling
            if num > len(dataset):
                num = len(dataset)
            batch = random.sample(range(len(dataset)), num)
            # save samples
            for idx, sample in enumerate(tqdm(batch, desc=f"Saving {mode} samples")):
                Input, Output = dataset[sample]
                Input_probability  = np.sum( Input[0,:,:,:,0]**2 +  Input[0,:,:,:,1]**2, axis=2)
                Output_probability = np.sum(Output[0,:,:,:,0]**2 + Output[0,:,:,:,1]**2, axis=2)
                # save input
                fig = plt.figure()
                plt.imshow(Input_probability, extent=[-40,40,-40,40])
                plt.title("Input probability")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.savefig(os.path.join(SAMPLES_DIR, f"sample_{idx}_input.png"))
                plt.close(fig)
                # save output
                fig = plt.figure()
                plt.imshow(Output_probability, extent=[-40,40,-40,40])
                plt.title("Output probability")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.savefig(os.path.join(SAMPLES_DIR, f"sample_{idx}_output.png"))
                plt.close(fig)

sampling_process_samples(SAVE_DIR, dataset_name, list(split_ratio.keys()), num=30)

# %%
