from data_qinn2 import get_wave_pair2d, propagation_distance, Lx, Ly, Nx, Ny
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

stepN = 50
batchN = 52
Epoch = 10000

def onehot(x: np.ndarray, start_dim=0):
    origin_shape = x.shape
    x = x.reshape(origin_shape[:start_dim] + (-1,))
    def onehot_(x):
        onehot = np.zeros_like(x)
        max_idx = np.argmax(x)
        onehot[max_idx] = 1
        return onehot
    onehot = np.apply_along_axis(onehot_, -1, x)
    return onehot.reshape(origin_shape)

def gumbels_simulate(intensitys: np.ndarray):
    BatchN, StepN, Nx, Ny = intensitys.shape
    probs = intensitys / intensitys.sum(axis=(-1, -2), keepdims=True)
    log_probs = np.log(probs + 1e-30)
    gumbels = np.random.gumbel(size=(BatchN, 1, Nx, Ny))
    gumbel_probs = log_probs + gumbels # (BatchN, StepN, Nx, Ny)
    return onehot(gumbel_probs, start_dim=-2)

_, o_waves = get_wave_pair2d(stepN)
o_intensitys: np.ndarray = o_waves[..., 0]**2 + o_waves[..., 1]**2
dz = propagation_distance / stepN

def generate_waves(batchN=32):
    intensitys = np.expand_dims(o_intensitys,0).repeat(batchN, axis=0)
    onehot_waves = gumbels_simulate(intensitys)

    return intensitys, onehot_waves

def get_move_distance(onehot_waves: np.ndarray):
    BatchN, StepN, Nx, Ny = onehot_waves.shape
    # get the xy position of one hot wave
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    XY = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1) # (Nx, Ny, 2)
    positions = np.tensordot(onehot_waves, XY, axes=((-2, -1), (0, 1))) # (BatchN, StepN, 2)
    # calculate the distance of each step
    distances = np.diff(positions, axis=-2) # (BatchN, StepN-1, 2)
    distances = np.square(distances).sum(axis=-1) + dz**2 # (BatchN, StepN-1)
    distances = np.sqrt(distances).sum(axis=-1, keepdims=True) # (BatchN, 1)
    # calculate the final position
    final_pos = positions[:, -1, :] # (BatchN, 2)
    return np.concatenate([final_pos, distances], axis=-1) # (BatchN, xyd)

def get_batch_distance(seed=0):
    np.random.seed(seed)
    _, onehot_waves = generate_waves(batchN)
    return get_move_distance(onehot_waves)

def main():
    # get the distance of each batch
    with Pool(24) as pool:
        async_results = pool.imap_unordered(get_batch_distance, range(Epoch))
        results = list(tqdm(async_results, total=Epoch))
        results = np.concatenate(results, axis=0) # (Epoch*batchN, 3)
    #results = get_batch_distance(0)

    # group the results if the final x is close to X
    static_step = 1e-5
    origin_results = results.copy()
    results = results[np.argsort(results[:, 0])] # sort by final x
    results[:, 0] = np.round(results[:, 0] / static_step) * static_step # round the final x
    uni_xs, idxs, counts = np.unique(results[:, 0], return_index=True, return_counts=True)
    groups = np.split(results, idxs, axis=0) # Gi, (Ni, 3)
    groups = groups[1:] # remove the first group
    mean_dis = np.zeros_like(uni_xs)
    for i, group in enumerate(groups):
        mean_dis[i] = group[:, -1].mean()

    # plot original intensity
    plt.imshow(o_intensitys[-1].transpose(), origin='lower')
    plt.colorbar()
    plt.savefig('test/original_intensity.png')
    plt.clf()

    # plot the distance distribution
    plt.hist(origin_results[:, 2], bins=100)
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.title('distance distribution')
    plt.savefig('test/distance_distribution.png')
    plt.clf()

    # plot the mean distance at each final x
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('final x')
    ax1.set_ylabel('mean distance', color=color)
    ax1.plot(uni_xs, mean_dis, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # plot the count at each final x
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('count', color=color)
    ax2.plot(uni_xs, counts, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.savefig('test/mean_distance_and_count.png')
    plt.clf()

    # plot the counts against final x in scatter
    plt.scatter(origin_results[:, 0], origin_results[:, 1], s=0.2, marker='.')
    plt.xlabel('final x')
    plt.ylabel('final y')
    plt.title('final position')
    plt.savefig('test/final_position.png')
    plt.clf()



if __name__ == '__main__':
    main()