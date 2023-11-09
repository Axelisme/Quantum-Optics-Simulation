
"""A script to generate data for the project."""
#%%
import os
import h5py
import random
import numpy as np
from typing import Callable
from util.io import clear_folder
from hyperparameters import base_conf, PROC_DATA_DIR
from data_qinn2 import get_wave_pair2d
import multiprocessing as mp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# some parameters
stepN = 10
TstepN = 10
dataset_name = f"S{stepN}T{TstepN}"
split_ratio = base_conf.split_ratio
SAVE_DIR = os.path.join(PROC_DATA_DIR, dataset_name)
clear_folder(SAVE_DIR) # clear the folder before generating data
random.seed(base_conf.seed)

#%%
def generate_process_data(dataset_path:str,
                          data_dtype:np.dtype,
                          data_loader:Callable,
                          mode_length:int = 1000,
                          max_batch_num:int = 1000) -> None:
    """
    Generate processed data for the project.
    """
    # save data to h5 file
    with mp.Pool(mp.cpu_count()) as pool:
        with h5py.File(dataset_path, mode='x') as writer:
            # write meta data
            writer.attrs["length"] = mode_length
            writer.attrs["data_dtype"] = str(data_dtype)
            # create dataset
            dataset = writer.create_dataset("dataset", (mode_length,), dtype=data_dtype)
            # write dataset
            saved_num = 0
            bar = tqdm(total=mode_length)
            while saved_num < mode_length:
                batch_num = min(max_batch_num, mode_length - saved_num)
                save_ids = list(range(saved_num, saved_num + batch_num))
                batch = pool.imap_unordered(data_loader, save_ids)
                for idx, data in zip(save_ids, batch):
                    dataset[idx] = data
                    bar.update()
                saved_num += batch_num
            bar.close()

def data_loader(_):
    Input,Ouput = get_wave_pair2d(TstepN)
    Input = Input.reshape(1,80,80,2)
    Ouput = Ouput[stepN,:,:,:]
    Ouput = Ouput.reshape(1,80,80,2)
    return Input,Ouput

data_dtype = np.dtype([("input", np.float32, (1,80,80,2)), ("output", np.float32,(1,80,80,2))])
data_length = 50000
for mode, ratio in split_ratio.items():
    DATASET_PATH = os.path.join(SAVE_DIR, f"{mode}.h5")
    mode_length = int(data_length * ratio)

    print(f"Writting {mode} dataset with length {mode_length} to {DATASET_PATH}")
    generate_process_data(DATASET_PATH, data_dtype, data_loader, mode_length)


#%%
def sampling_process_samples(dataset_path:str,
                             sample_saver:Callable,
                             max_num:int = 20,
                             *args, **kwargs) -> None:
    """
    Sampling some samples from the processed data.
    """
    with h5py.File(dataset_path, mode='r') as reader:
        # load data
        dataset = reader["dataset"]
        data_length = reader.attrs["length"]
        # sampling
        sample_ids = random.sample(range(data_length), max_num)
        # save samples
        for idx, sample_id in enumerate(tqdm(sample_ids)):
            sample = dataset[sample_id]
            sample_saver(idx, sample, *args, **kwargs)

def sample_saver(id, sample, sample_dir):
    Input, Output = sample
    Input_probability  = Input[0,:,:,0]**2 +  Input[0,:,:,1]**2
    Output_probability = Output[0,:,:,0]**2 + Output[0,:,:,1]**2
    # save input
    fig = plt.figure()
    plt.imshow(Input_probability, extent=[-40,40,-40,40])
    plt.colorbar()
    plt.title("Input probability")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(sample_dir, f"sample_{id}_input.png"))
    plt.close(fig)
    # save output
    fig = plt.figure()
    plt.imshow(Output_probability, extent=[-40,40,-40,40])
    plt.colorbar()
    plt.title("Output probability")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(sample_dir, f"sample_{id}_output.png"))
    plt.close(fig)

for mode in split_ratio.keys():
    dataset_path = os.path.join(SAVE_DIR, f"{mode}.h5")
    SAMPLES_DIR = os.path.join(SAVE_DIR, f"{mode}_samples")
    clear_folder(SAMPLES_DIR)
    sampling_process_samples(dataset_path, sample_saver, sample_dir=SAMPLES_DIR)

# %%
