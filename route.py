from data_qinn2 import get_wave_pair
import matplotlib.pyplot as plt
import numpy as np

iwave, pwave = get_wave_pair()
iwave_intensity = iwave[..., 0]**2 + iwave[..., 1]**2
pwave_intensity = pwave[..., 0]**2 + pwave[..., 1]**2

assert iwave_intensity.shape == pwave_intensity[0].shape, f"iwave.shape: {iwave_intensity.shape}, pwave.shape: {pwave_intensity.shape}"
StepN, Nx, Ny, Nz = pwave_intensity.shape

def softmax(x, tau=1.0):
    """Compute softmax values for each sets of scores in x."""
    x = x / tau
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Convert pwave to probability distribution by gumbel softmax
gumbels = np.random.gumbel(size=(Nx, Ny, Nz))
prob_pwave = pwave_intensity * Nx * Ny * Nz
prob_pwave = prob_pwave + gumbels
prob_pwave = prob_pwave / np.sum(prob_pwave, axis=(1,2,3), keepdims=True)
onehot_pwave = np.zeros_like(prob_pwave)
for i in range(StepN):
    onehot_pwave[i] = softmax(prob_pwave[i], tau=0.01)

# print probablity along step in xz plane
origin_xz_intensity = np.sum(pwave_intensity, axis=(1,3)) # (StepN, Nx, Ny, Nz) -> (StepN, Ny)
plt.figure()
plt.imshow(origin_xz_intensity)
plt.colorbar()
plt.show()
plt.savefig('test/origin_xz_intensity.png')
plt.close()

pwave_xz_intensity = np.sum(prob_pwave, axis=(2,3)) # (StepN, Nx, Ny, Nz) -> (StepN, Nx)
plt.figure()
plt.imshow(pwave_xz_intensity)
plt.colorbar()
plt.show()
plt.savefig('test/pwave_xz_intensity.png')
plt.close()

onehot_pwave_xz_intensity = np.sum(onehot_pwave, axis=(2,3)) # (StepN, Nx, Ny, Nz) -> (StepN, Nx)
plt.figure()
plt.imshow(onehot_pwave_xz_intensity)
plt.colorbar()
plt.show()
plt.savefig('test/onehot_pwave_xz_intensity.png')
plt.close()