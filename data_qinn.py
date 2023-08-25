import numpy as np
import random
import math
import time
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# Physical constants
mu0 = 2.0e-7 * np.pi    # free space magnetic permeability [Vs/Am]
c0  = 2.99792458e+8     # free space light speed [m/s]
h_bar    = 0.5*1.054*10**-34
epsilon0 = 1.0 / (mu0 * c0**2)         # free space permittivity [As/Vm]
eta0     = np.sqrt(mu0 / epsilon0)     # free space impedance [ohm]

# Physical parameters
n = 1.0                 # refractive index of the medium
L0 = 532e-7             # wave function specific space length [m]
Lx, Ly = L0, L0
Lz = 532e-8            # wave function specific space length in z direction [m]
lambda0 = 532e-9        # free space wavelength [m]
propagation_distance = 0.19e-3  # propagation distance [m]

# Derived parameters
k0 = 2*np.pi / lambda0       	       # free space wavenumber [m-1]
k  = n * k0                            # medium wavenumber [m-1]

# Computational domain parameters
N = 80                 # number of spatial points
Nx, Ny, Nz = N, N, N   # number of spatial points in x, y, z directions
stepN = 10             # number of temporal points
seam_width_pixels = 2
seam1_pos = round(1.8*Nx/4)
seam2_pos = round(2.2*Nx/4)


def get_gaussian_wave_packet(N3, L3, r3, k3, sigma3) -> np.ndarray:
    # parameters
    Nx, Ny, Nz = N3
    Lx, Ly, Lz = L3
    x0, y0, z0 = r3
    kx, ky, kz = k3
    sigma_x, sigma_y, sigma_z = sigma3

    # define one dimension wave function
    def one_dimension_wave_func(X, x0, k, sigma):
        constant = 1/math.sqrt(sigma*math.sqrt(math.pi))
        return constant * np.exp(-((X-x0)**2)/(2*sigma**2)) * np.exp(1j*k*X)

    # 3d grid
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    z = np.linspace(-Lz/2, Lz/2, Nz)
    X, Y, Z = np.meshgrid(x, y, z)

    # 3d wave function
    wave = one_dimension_wave_func(X, x0, kx, sigma_x) * \
            one_dimension_wave_func(Y, y0, ky, sigma_y) * \
            one_dimension_wave_func(Z, z0, kz, sigma_z)
    wave = np.stack((wave.real, wave.imag), axis=-1)

    return wave

def get_slits_mask(N2, seam1_pos, seam2_pos, seam_width_pixels):

    # N2 = (Nx, Ny)
    Nx, _ = N2
    # 第一個狹縫
    S1w_s = seam1_pos - round(seam_width_pixels/2)
    S1w_e = seam1_pos + round(seam_width_pixels/2)
    # 第二個狹縫
    S2w_s = seam2_pos - round(seam_width_pixels/2)
    S2w_e = seam2_pos + round(seam_width_pixels/2)
    # 狹縫高度起始位置
    Hs = round(0.5*Ny/4)
    He = round(3.5*Ny/4)

    # 製作狹縫
    matrix = np.zeros(N2, dtype=int)
    matrix[S1w_s:S1w_e, Hs:He] = 1
    matrix[S2w_s:S2w_e, Hs:He] = 1

    return matrix

def propagate(Input: np.ndarray,
                L3: Tuple[float, float, float],
                k: float,
                stepN: int,
                propagation_distance: float,
                index_potential: Optional[np.ndarray] = None,
                absorbing_boundary: bool = True,
                paraxial: bool = True) -> np.ndarray:
    assert Input.ndim == 4, "Input.ndim must be 4"
    assert Input.shape[3] == 2, "Input.shape must be equal to (Nx, Ny, Nz, 2)"
    assert stepN > 0, "stepN must be greater than 0"
    assert len(L3) == 3, "L3 must be (Lx, Ly, Lz)"

    # parameters
    Nx, Ny, Nz, _ = Input.shape
    Lx, Ly, Lz = L3
    dx = Lx / Nx  # discretization step in x
    dy = Ly / Ny  # discretization step in y
    dz = Lz / Nz  # discretization step in z
    z_step = propagation_distance / stepN

    # wave vector discretization
    dkx = 2*np.pi / Lx
    dky = 2*np.pi / Ly
    kx = dkx * np.concatenate((np.arange(0,Nx/2,1), np.arange(-Nx/2,0,1)))   # spatial frequencies vector in the x direction (swapped)                                                         # discretization in the spatial spectral domain along the y direction
    ky = dky * np.concatenate((np.arange(0,Ny/2,1), np.arange(-Ny/2,0,1)))   # spatial frequencies vector in the y direction (swapped)
    [Kx, Ky] = np.meshgrid(kx, ky)
    K2 = np.multiply(Kx,Kx) + np.multiply(Ky,Ky)    # Here we define some variable so that we don't need to compute them again and again

    # index potential
    if index_potential is None:
        index_potential = np.ones((Nx, Ny), dtype=float)
    else:
        assert index_potential.shape == (Nx, Ny), "V.shape must be equal to (Nx, Ny)"

    # Absorbing boundary
    if absorbing_boundary:
        x = dx * np.arange(-Nx/2,Nx/2,1)  # normalized x dimension vector
        y = dy * np.arange(-Ny/2,Ny/2,1)  # normalized x dimension vector
        [X, Y] = np.meshgrid(x, y)
        super_gaussian = np.exp(-((X / (0.9*Lx/(2*np.sqrt(np.log(2)))) )**20 + (Y / (0.9*Ly/(2*np.sqrt(np.log(2)))) )**20))
    else:
        super_gaussian = None

    # convert the input to complex wave form
    phi0 = Input[:,:,:,0] + 1j * Input[:,:,:,1]

    # initialize the output
    Output = np.array(np.zeros((stepN+1,Nx,Ny,Nz,2), dtype=float))

    # Propagation
    for Layer in range(Nz):
        u0 = phi0[:,:,Layer]
        if absorbing_boundary:
            u0 = u0 * super_gaussian
        fields = np.zeros((stepN+1, Nx, Ny), dtype=complex)
        fields[0] = u0
        for step_i in range(1, stepN+1):
            if paraxial:
                ## paraxial
                u1 = np.fft.ifft2(np.fft.fft2(u0) * np.exp(-1j * K2 / (2*k) * 0.5 * z_step))	# First linear half step
                u2 = u1 * np.exp(1j * z_step * (index_potential))                               # Refraction step
                u3 = np.fft.ifft2(np.fft.fft2(u2) * np.exp(-1j * K2 / (2*k) * 0.5 * z_step))	# Second linear step
            else:
                ## Nonparaxial code
                u1 = np.fft.ifft2(np.fft.fft2(u0) * np.exp(-1j * K2 * 0.5 * z_step / (k + np.sqrt(k**2 - K2))))
                u2 = u1 * np.exp(1j * z_step * (index_potential))
                u3 = np.fft.ifft2(np.fft.fft2(u2) * np.exp(-1j * K2 * 0.5 * z_step / (k + np.sqrt(k**2 - K2))))

            ## Absorbing boundary
            u0 = u3
            if absorbing_boundary:
                u0 = u0 * super_gaussian

            ## Save to the field
            fields[step_i] = u0
        # Save this layer
        Output[:,:,:,Layer,0] = np.real(fields)
        Output[:,:,:,Layer,1] = np.imag(fields)

    return Output

def get_wave_pair(stepN) -> Tuple[np.ndarray, np.ndarray]:
    # 調製參數
    x0 = random.uniform(-Lx/3, Lx/3)      # x方向波包中心位置
    y0 = random.uniform(-Ly/3, Ly/3)      # y方向波包中心位置
    z0 = random.uniform(-Lz/3, Lz/3)      # z方向波包中心位置
    r3 = (x0, y0, z0)
    theta = random.uniform(0, np.pi/2) # 波包動量與z軸夾角
    phi = random.uniform(0, 2*np.pi)      # 波包動量在xy平面旋轉角度
    kx = k * np.sin(theta) * np.cos(phi)  # x方向波數
    ky = k * np.sin(theta) * np.sin(phi)  # y方向波數
    kz = k * np.cos(theta)                # z方向波數
    k3 = (kx, ky, kz)
    sigma_x = random.uniform(Lx/8, 2*Lx/10)    # 高斯分佈標準差
    sigma_y = random.uniform(Ly/8, 2*Ly/10)    # 高斯分佈標準差
    sigma_z = random.uniform(Lz/8, 2*Lz/10)    # 高斯分佈標準差
    sigma3 = (sigma_x, sigma_y, sigma_z)

    # 計算波包
    N3 = (Nx, Ny, Nz)
    L3 = (Lx, Ly, Lz)
    Input = get_gaussian_wave_packet(N3, L3, r3, k3, sigma3)

    # 加上夾縫遮罩
    slit_mask = get_slits_mask((Nx, Ny), seam1_pos, seam2_pos, seam_width_pixels)
    slit_mask = slit_mask.reshape((Nx, Ny, 1, 1))
    slit_mask = np.tile(slit_mask, (1, 1, Nz, 2))
    Slit_input = Input * slit_mask

    # 計算傳遞
    Output = propagate(Slit_input, L3, k, stepN, propagation_distance/10, paraxial=False, absorbing_boundary=True)

    # Normalize the intensity
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    norm_factor = math.sqrt(dx * dy * dz)
    Input = Input * norm_factor
    Output = Output * norm_factor

    return Input, Output

def main():
    # 設定種子
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # 計算波包與傳遞
    t_start = time.time()
    Input, Output = get_wave_pair(stepN)
    t_end = time.time()
    print(f"time elapsed: {t_end-t_start:.2f} seconds")

    # convert wave form to intensity
    Input_intensity = Input[:,:,:,0]**2 + Input[:,:,:,1]**2
    Output_intensity = Output[:,:,:,:,0]**2 + Output[:,:,:,:,1]**2
    print(f"Input shape: {Input.shape}")
    print(f"Input sum: {np.sum(Input_intensity)}")
    print(f"Output shape: {Output.shape}")
    print(f"Output sum: {np.sum(Output_intensity[-1])}")

    # 畫出波包與傳遞後的波包
    fig_dpi = 80
    plt.figure(dpi = fig_dpi)
    input_yx_intensity = np.sum(Input_intensity, axis=2)
    plt.imshow(input_yx_intensity, extent=[-Ly/2*1e3,Ly/2*1e3,-Lx/2*1e3,Lx/2*1e3])
    plt.colorbar()
    plt.xlabel('y axis [mm]')
    plt.ylabel('x axis [mm]')
    plt.title('Input abs', fontsize='x-large')
    plt.show()
    plt.savefig('Input.png')

    plt.figure(dpi = fig_dpi)
    output_yx_intensity = np.sum(Output_intensity, axis=3)
    plt.imshow(output_yx_intensity[0], extent=[-Ly/2*1e3,Ly/2*1e3,-Lx/2*1e3,Lx/2*1e3])
    plt.colorbar()
    plt.xlabel('y axis [mm]')
    plt.ylabel('x axis [mm]')
    plt.title('Input slits abs', fontsize='x-large')
    plt.show()
    plt.savefig('Input_slits.png')

    plt.figure(dpi = fig_dpi)
    plt.imshow(output_yx_intensity[-1], extent=[-Ly/2*1e3,Ly/2*1e3,-Lx/2*1e3,Lx/2*1e3])
    plt.colorbar()
    plt.xlabel('y axis [mm]')
    plt.ylabel('x axis [mm]')
    plt.title('Output abs', fontsize='x-large')
    plt.show()
    plt.savefig('Output.png')

    plt.figure(dpi = fig_dpi)
    output_xp_intensity = np.sum(Output_intensity, axis=(2,3))
    plt.imshow(output_xp_intensity, extent=[-Lx/2*1e3,Lx/2*1e3,propagation_distance*1e3,0])
    plt.colorbar()
    plt.xlabel('x axis [mm]')
    plt.ylabel('z axis [mm]')
    plt.title('Propagation abs', fontsize='x-large')
    plt.show()
    plt.savefig('Propagation.png')

    print('Finish')


if __name__ == '__main__':
    main()