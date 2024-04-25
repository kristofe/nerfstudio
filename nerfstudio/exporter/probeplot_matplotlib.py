import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, hsv_to_rgb
from scipy.special import sph_harm


def sph_harm_real(m, l, phi, theta):
    #TODO: CONFIRM THIS IS CORRECT
    f = sph_harm(m, l, theta, phi)
    
    if m < 0:
        f = np.sqrt(2) * (-1)**m * f.imag
    elif m > 0:
        f = np.sqrt(2) * (-1)**m * f.real
    return f

def calculate_total_sh(coeffs, normalized=False):
    ls = [0,1,1,1]
    ms = [0,-1,0,1]
    color = 0
    f = None
    for i in range(4):
        l = ls[i]
        m = ms[i]
        coeff = coeffs[color, i]
        #print(f"l={l}, m={m} color={color} coeff={coeff}")
        sp = sph_harm_real(m, l, theta, phi)
        #print(f"min: {sp.min()}, max: {sp.max()}")
        sp *= coeff

        if f is None:
            f = sp
        else:
            f += sp
        
        #print(f"min: {f.min()}, max: {f.max()}")
    f = f.real
    # Normalize the colors
    if normalized:
        f = (f - f.min()) / (f.max() - f.min())
        print(f"normalized: min: {f.min()}, max: {f.max()}")

    return f

# Spherical coordinates
phi = np.linspace(0, np.pi, 32)  # Polar angle
theta = np.linspace(0, 2 * np.pi, 32)  # Azimuthal angle
phi, theta = np.meshgrid(phi, theta)

# current dataset only has DC and l1  or l0 and l1 bands of SH... so just 4 per channel.
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)



#probegrid = ProbeGrid("mono/usd/scene-000000-7.direct.0304641882000459.usd")
import os

probegrid = np.load("experiments/exports/pcd/sh_coeffs.npz")
# self.grid_tensor = torch.Tensor(self.coefficients.reshape((3, self.res[2], self.res[1], self.res[0] * 4)))
#print(probegrid.grid_tensor.shape)
#print(probegrid.coefficients.shape)
#print(probegrid.data_validity.shape)
#print(probegrid.res.shape)
points = probegrid['points']
sh_coeffs = probegrid['sh_coeffs']
grid_resolution = probegrid['grid_resolution']
print(f'points: {points.shape}, sh_coeffs: {sh_coeffs.shape}')

'''
res = probegrid.res
shape = probegrid.grid_tensor.shape
channels = probegrid.coefficients.shape[0]
coefficient_count = probegrid.coefficients.shape[2]
#print(f'grid shape: {res}, number of color channels: {channels}, number of coefficients per channel: {coefficient_count}')
coefficients_grid_layout = probegrid.coefficients.reshape((channels, res[2], res[1], res[0], coefficient_count))
#print(f'coefficients_grid_layout.shape: {coefficients_grid_layout.shape}')
'''

grid_x = 8 #res[0]
grid_y = 8 #res[1]

# setup figure/plot grid
figsize_px, DPI = 800, 100
figsize_in = figsize_px / DPI
fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=DPI)
spec = gridspec.GridSpec(ncols=grid_x, nrows=grid_y, figure=fig)

for yy in range(grid_y):
    for xx in range(grid_x):
        coeffs = sh_coeffs[:,0,yy,xx,:]
        #f = calculate_total_sh(coeffs, normalized=True)

        #TODO: CONFIRM THIS IS CORRECT
        h = calculate_total_sh(coeffs[0:1,], normalized=True)
        s = calculate_total_sh(coeffs[1:2:,], normalized=True)
        v = calculate_total_sh(coeffs[2:3,], normalized=True)
        #have to map back into rgb
        hsv = np.stack((h, s, v), axis=-1)
        rgb = hsv_to_rgb(hsv)

        ax = fig.add_subplot(spec[xx, yy], projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
        # Hide axes
        ax.set_axis_off()

plt.tight_layout()
plt.savefig('probegrid.png')
#plt.show()
