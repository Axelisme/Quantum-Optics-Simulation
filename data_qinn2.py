#%% SLit and Scaling setting
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
n = 1.0                            # refractive index of the medium
L0 = 1.0982e-3#5e-3                         # wave function specific space length [m]
Lx, Ly, Lz = L0, L0/4, L0
lambda0 = 532e-9                   # free space wavelength [m]
seam_width = 1.5244e-5#4.1e-5
seam_distance = 9.1463e-5#3.66e-4
propagation_distance = 1.5724e-2#2.52e-1      # propagation distance [m]

# Derived parameters
k0 = 2*np.pi / lambda0       	       # free space wavenumber [m-1]
k  = n * k0                            # medium wavenumber [m-1]

# Computational domain parameters
N = 256                 # number of spatial points
Nx, Ny, Nz = N, N//4, N   # number of spatial points in x, y, z directions
stepN = 25             # number of temporal points
seam_width_pixels = max(round(seam_width/L0*Nx), 2)
seam_distance_pixels = max(round(seam_distance/L0*Nx), 2)
seam1_pos = (Nx - seam_distance_pixels) // 2
seam2_pos = (Nx + seam_distance_pixels) // 2
print(f"seam_width_pixels: {seam_width_pixels}")
print(f"seam_distance_pixels: {seam_distance_pixels}")
print(f"seam1_pos: {seam1_pos}")
print(f"seam2_pos: {seam2_pos}")



def get_gaussian_wave_packet2d(N2, L2, k) -> np.ndarray:
    # parameters
    Nx, Ny = N2
    Lx, Ly = L2
    a=0.12

    x0 = random.uniform(-(a*1e-4), (a*1e-4))      # x方向波包中心位置
    y0 = random.uniform(-(a*1e-4), (a*1e-4))      # y方向波包中心位置

    #theta = random.uniform(0, np.pi/32) # 波包動量與z軸夾角
    #phi = random.uniform(0, 2*np.pi)      # 波包動量在xy平面旋轉
    theta = 0
    phi = 0

    kx = k * np.sin(theta) * np.cos(phi)  # x方向波數
    ky = k * np.sin(theta) * np.sin(phi)  # y方向波數

    sigma_x = random.uniform(Lx/2, Lx)    # 高斯分佈標準差
    sigma_y = random.uniform(Lx/2, Lx)    # 高斯分佈標準差

    dx = Lx/Nx
    dy = Ly/Ny

    # define one dimension wave function
    def one_dimension_wave_func(X, x0, k, sigma):
        constant = 1/math.sqrt(sigma*math.sqrt(math.pi))
        return constant * np.exp(-((X-x0)**2)/(2*sigma**2)) * np.exp(1j*k*X)

    # 3d grid
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 3d wave function
    wave = one_dimension_wave_func(X, x0, kx, sigma_x) * \
            one_dimension_wave_func(Y, y0, ky, sigma_y)
    wave *= math.sqrt(dx*dy) # normalization
    wave = np.stack((wave.real, wave.imag), axis=-1)

    return wave

def get_gaussian_wave_packet3d(N3, L3, k) -> np.ndarray:
    # parameters
    Nx, Ny, Nz = N3
    Lx, Ly, Lz = L3
    a=0.12

    x0 = random.uniform(-(a*1e-4), (a*1e-4))      # x方向波包中心位置
    y0 = random.uniform(-(a*1e-4), (a*1e-4))      # y方向波包中心位置
    z0 = random.uniform(-(a*1e-4), (a*1e-4))      # z方向波包中心位置

    theta = random.uniform(0, np.pi/8) # 波包動量與z軸夾角
    phi = random.uniform(0, 2*np.pi)      # 波包動量在xy平面旋轉

    kx = k * np.sin(theta) * np.cos(phi)  # x方向波數
    ky = k * np.sin(theta) * np.sin(phi)  # y方向波數
    kz = k * np.cos(theta)                # z方向波數


    Naa=0.006 # Boundary (Min)
    Nab=0.007 # Boundary (MAx)

    sigma_px = random.uniform(Naa*1e-27, Nab*1e-27)    # 高斯分佈標準差
    sigma_py = random.uniform(Naa*1e-27, Nab*1e-27)    # 高斯分佈標準差
    sigma_pz = random.uniform(Naa*1e-27, Nab*1e-27)

    sigma_x = h_bar/sigma_px
    sigma_y = h_bar/sigma_py
    sigma_z = h_bar/sigma_pz

    dx = Lx/Nx
    dy = Ly/Ny
    dz = Lz/Nz

    # define one dimension wave function
    def one_dimension_wave_func(X, x0, k, sigma):
        constant = 1/math.sqrt(sigma*math.sqrt(math.pi))
        return constant * np.exp(-((X-x0)**2)/(2*sigma**2)) * np.exp(1j*k*X)

    # 3d grid
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    z = np.linspace(-Lz/2, Lz/2, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 3d wave function
    wave = one_dimension_wave_func(X, x0, kx, sigma_x) * \
            one_dimension_wave_func(Y, y0, ky, sigma_y) * \
            one_dimension_wave_func(Z, z0, kz, sigma_z)
    wave *= math.sqrt(dx*dy*dz) # normalization
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

    # 製作狹縫
    matrix = np.zeros(N2, dtype=int)
    matrix[S1w_s:S1w_e, :] = 1
    matrix[S2w_s:S2w_e, :] = 1

    return matrix

def propagate2d(Input: np.ndarray,
                L2: Tuple[float, float],
                k: float,
                stepN: int,
                propagation_distance: float,
                index_potential: Optional[np.ndarray] = None,
                absorbing_boundary: bool = True,
                paraxial: bool = True) -> np.ndarray:
    assert Input.ndim == 3, "Input.ndim must be 3"
    assert Input.shape[2] == 2, "Input.shape must be equal to (Nx, Ny, 2)"
    assert stepN > 0, "stepN must be greater than 0"
    assert len(L2) == 2, "L2 must be (Lx, Ly)"

    # parameters
    Nx, Ny, _ = Input.shape
    Lx, Ly = L2
    dx = Lx / Nx  # discretization step in x
    dy = Ly / Ny  # discretization step in y
    z_step = propagation_distance / stepN

    # wave vector discretization
    dkx = 2*np.pi / Lx
    dky = 2*np.pi / Ly
    kx = dkx * np.concatenate((np.arange(0,Nx/2,1), np.arange(-Nx/2,0,1)))   # spatial frequencies vector in the x direction (swapped)                                                         # discretization in the spatial spectral domain along the y direction
    ky = dky * np.concatenate((np.arange(0,Ny/2,1), np.arange(-Ny/2,0,1)))   # spatial frequencies vector in the y direction (swapped)
    [Kx, Ky] = np.meshgrid(kx, ky, indexing='ij')
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
        [X, Y] = np.meshgrid(x, y, indexing='ij')
        super_gaussian = np.exp(-((X / (0.9*Lx/(2*np.sqrt(np.log(2)))) )**20 + (Y / (0.9*Ly/(2*np.sqrt(np.log(2)))) )**20))
    else:
        super_gaussian = None

    # convert the input to complex wave form
    phi0 = Input[:,:,0] + 1j * Input[:,:,1]

    # initialize the output
    Output = np.array(np.zeros((stepN+1, Nx, Ny ,2), dtype=float))

    # Propagation
    u0 = phi0
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

    Output[:,:,:,0] = np.real(fields)
    Output[:,:,:,1] = np.imag(fields)

    return Output

def propagate3d(Input3d: np.ndarray,
                L3: Tuple[float, float, float],
                k: float,
                stepN: int,
                propagation_distance: float,
                index_potential: Optional[np.ndarray] = None,
                absorbing_boundary: bool = True,
                paraxial: bool = True) -> np.ndarray:
    assert Input3d.ndim == 4, "Input.ndim must be 4"
    assert Input3d.shape[3] == 2, "Input.shape must be equal to (Nx, Ny, Nz, 2)"
    assert stepN > 0, "stepN must be greater than 0"
    assert len(L3) == 3, "L3 must be (Lx, Ly, Lz)"

    # convert the input to complex wave form
    phi0 = Input3d[:,:,:,0] + 1j * Input3d[:,:,:,1]

    # initialize the output
    Output = np.array(np.zeros((stepN+1,Nx,Ny,Nz,2), dtype=float))

    # Propagation
    for Layer in range(Nz):
        Input2d = Input3d[:,:,Layer,:]
        Output2d = propagate2d(Input2d, L3[:2], k, stepN, propagation_distance, index_potential, absorbing_boundary, paraxial)
        Output[:,:,:,Layer,:] = Output2d

    return Output


#%%

def get_wave_pair2d(stepN:int = stepN) -> Tuple[np.ndarray, np.ndarray]:
    # 計算波包
    N2 = (Nx, Ny)
    L2 = (Lx, Ly)
    k1 = k
    Input = get_gaussian_wave_packet2d(N2, L2, k1)

    # 加上夾縫遮罩
    slit_mask = get_slits_mask((Nx, Ny), seam1_pos, seam2_pos, seam_width_pixels)
    slit_mask = slit_mask.reshape((Nx, Ny, 1))
    slit_mask = np.tile(slit_mask, (1, 1, 2))
    Slit_input = Input * slit_mask

    # 計算傳遞
    Output = propagate2d(Slit_input, L2, k, stepN, propagation_distance, paraxial=False, absorbing_boundary=True)

    return Input, Output

def get_wave_pair3d() -> Tuple[np.ndarray, np.ndarray]:

    # 計算波包
    N3 = (Nx, Ny, Nz)
    L3 = (Lx, Ly, Lz)
    k1 = k * random.uniform(0.5, 2)
    Input = get_gaussian_wave_packet3d(N3, L3, k1)

    # 加上夾縫遮罩
    slit_mask = get_slits_mask((Nx, Ny), seam1_pos, seam2_pos, seam_width_pixels)
    slit_mask = slit_mask.reshape((Nx, Ny, 1, 1))
    slit_mask = np.tile(slit_mask, (1, 1, Nz, 2))
    Slit_input = Input * slit_mask

    # 計算傳遞
    Output = propagate3d(Slit_input, L3, k, stepN, propagation_distance, paraxial=False, absorbing_boundary=True)

    return Input, Output


#%
def main3d():
    # 設定種子
    #seed = 10
    #np.random.seed(seed)
    #random.seed(seed)

    # 計算波包與傳遞
    t_start = time.time()
    Input, Output = get_wave_pair3d()

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


    Total = sum(sum(sum(Input_intensity)))
    print('Nomalized.{}',Total)
    print(Output.size)
    print('Finish')


def main2d():
    # 設定種子
    #seed = 10
    #np.random.seed(seed)
    #random.seed(seed)

    # 計算波包與傳遞
    t_start = time.time()
    Input, Output = get_wave_pair2d()
    t_end = time.time()
    print(f"time elapsed: {t_end-t_start:.2f} seconds")

    # convert wave form to intensity
    Input_intensity = Input[:,:,0]**2 + Input[:,:,1]**2
    Output_intensity = Output[:,:,:,0]**2 + Output[:,:,:,1]**2
    print(f"Input shape: {Input.shape}")
    print(f"Input sum: {np.sum(Input_intensity)}")
    print(f"Output shape: {Output.shape}")
    print(f"Output sum: {np.sum(Output_intensity[-1])}")

    # 畫出波包與傳遞後的波包
    fig_dpi = 80
    plt.figure(dpi = fig_dpi)
    input_yx_intensity = Input_intensity
    plt.imshow(input_yx_intensity, extent=[-Ly/2*1e3,Ly/2*1e3,-Lx/2*1e3,Lx/2*1e3])
    plt.colorbar()
    plt.xlabel('y axis [mm]')
    plt.ylabel('x axis [mm]')
    plt.title('Input abs', fontsize='x-large')
    plt.show()
    plt.savefig('Input.png')

    plt.figure(dpi = fig_dpi)
    output_yx_intensity = Output_intensity
    plt.imshow(output_yx_intensity[0], extent=[-Ly/2*1e3,Ly/2*1e3,-Lx/2*1e3,Lx/2*1e3])
    plt.colorbar()
    plt.xlabel('y axis')
    plt.ylabel('x axis')
    plt.title('Input slits abs', fontsize='x-large')
    plt.show()
    plt.savefig('Input_slits.png')

    plt.figure(dpi = fig_dpi)
    plt.imshow(output_yx_intensity[-1], extent=[-Ly/2*1e3,Ly/2*1e3,-Lx/2*1e3,Lx/2*1e3])
    plt.colorbar()
    plt.xlabel('y axis')
    plt.ylabel('x axis')
    plt.title('Output abs', fontsize='x-large')
    plt.show()
    plt.savefig('Output.png')

    plt.figure(dpi = fig_dpi)
    output_xp_intensity = np.sum(Output_intensity, axis=2)
    plt.imshow(output_xp_intensity, extent=[-Lx/2*1e3,Lx/2*1e3,propagation_distance*1e3,0])
    plt.colorbar()
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    plt.title('Propagation abs', fontsize='x-large')
    plt.show()
    plt.savefig('Propagation.png')


    Total = sum(sum(Input_intensity))
    print('Nomalized.',Total)
    print(Output.size)
    print('Finish')

if __name__ == '__main__':
    main2d()

