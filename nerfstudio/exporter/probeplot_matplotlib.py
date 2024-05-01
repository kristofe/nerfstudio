import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from exporter_utils import project_sh, sample_sphere_low_discrepancy
from matplotlib.colors import LightSource, hsv_to_rgb
from scipy.special import sph_harm


def direction_to_color(direction):
    # Normalize the direction vector to ensure it's a unit vector
    direction = direction / torch.norm(direction, p=2)

    # Calculate azimuth and elevation
    azimuth = torch.atan2(direction[1], direction[0])  # Range from -pi to pi
    elevation = torch.asin(direction[2])  # Range from -pi/2 to pi/2

    # Normalize azimuth from 0 to 1 (for hue in HSV)
    hue = (azimuth + torch.pi) / (2 * torch.pi)

    # Normalize elevation from 0 to 1 (for value in HSV)
    # Assuming the color is brightest at the equator and dark at the poles
    value = torch.cos(elevation)  # cos(elevation) ranges from 0 at poles to 1 at equator

    # Convert HSV to RGB
    hsv = np.asarray([hue.item(), 1.0, value.item()])  # Full saturation, variable value

    #have to map back into rgb
    #hsv = np.stack((h, s, v), axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Convert to RGB values between 0 and 255
    #rgb = tuple(int(255 * x) for x in rgb)
    return rgb

def create_test_colors(view_directions, grid_res):
    # color is based on the x component of the direction
    rgbs = view_directions.clone()  
    rgbs[:,1:3] = rgbs[:,0:1]
    rgbs = (rgbs + 1.0) / 2.0
    rgbs = rgbs* 0.0 + 1.0
    return rgbs


def create_sh_test_data(grid_res = 1, device = torch.device('cuda'), num_directions = 1024):
    results = []
    linepoints = torch.linspace(-1, 1, grid_res, device=device)
    x,y,z = torch.meshgrid(linepoints, linepoints, linepoints, indexing='ij')
    positions = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    num_rays = positions.shape[0]
    sh_coeffs = torch.zeros((num_rays, 3, 4), device=device)

    sh_indices = [[0,0,0], [1, 1,-1], [2, 1,0], [3, 1,1]]

    random_directions = sample_sphere_low_discrepancy(device, num_directions)
    for rand_direction in random_directions:
        view_directions = torch.ones((num_rays, 3), device=device)
        view_directions = view_directions * rand_direction



        rgbs = create_test_colors(view_directions, grid_res)#THIS IS WHERE WE GIVE A KNOWN COLOR DISTRIBUTION 

        #Calculate the SH Coefficients
        for idx, l, m in sh_indices:
            sh = project_sh(L = l, M = m, n = view_directions) 
            for color_idx in range(3):
                sh_coeffs[:,color_idx,idx] += sh * rgbs[:,color_idx]

        results.append(rgbs)
    rgbs_mean = torch.stack(results, dim=0).mean(dim=0)

    sh_coeffs = sh_coeffs / num_directions
    sh_coeffs[:,:,1:] = 0

    probegrid = {'points': positions.cpu().numpy(), 'sh_coeffs': sh_coeffs.cpu().numpy(), 'grid_resolution': (grid_res, grid_res, grid_res)} 
    return probegrid

def sph_harm_real(m, l, phi, theta):
    #TODO: CONFIRM THIS IS CORRECT
    f = sph_harm(m, l, theta, phi)
    
    #if m < 0:
    #    f = np.sqrt(2) * (-1)**m * f.imag
    #elif m >= 0:
    #    f = np.sqrt(2) * (-1)**m * f.real
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
        #sp = sph_harm_real(m, l, theta, phi)
        sp = sph_harm(m, l, theta, phi)
        sp = np.abs(sp)
        #print(f"min: {sp.min()}, max: {sp.max()}")
        sp *= coeff

        if f is None:
            f = sp
        else:
            f += sp
        
        print(f"min: {f.min()}, max: {f.max()}")
    #f = f.real

    # Normalize the colors
    #if normalized:
    #    f = (f - f.min()) / (f.max() - f.min())
    #    print(f"normalized: min: {f.min()}, max: {f.max()}")

    return f

# Spherical coordinates
phi = np.linspace(0, np.pi, 32)  # Polar angle
theta = np.linspace(0, 2 * np.pi, 32)  # Azimuthal angle
phi, theta = np.meshgrid(phi, theta)

# current dataset only has DC and l1  or l0 and l1 bands of SH... so just 4 per channel.
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)



import os

# BE CAREFUL!!! WHAT COLOR SPACE ARE THE COEFFICENTS IN?  RGB or HSV?
#probegrid = ProbeGrid("mono/usd/scene-000000-7.direct.0304641882000459.usd") # HSV COLOR SPACE
probegrid = np.load("experiments/exports/pcd/sh_coeffs.npz") # RGB Color Space
#probegrid = create_sh_test_data(grid_res = 1, device = torch.device('cpu'), num_directions = 1024)
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

grid_x = grid_resolution[0]
grid_y = grid_resolution[1]

sh_coeffs = sh_coeffs.reshape((grid_resolution[0], grid_resolution[1], grid_resolution[2],3,4))

# setup figure/plot grid
figsize_px, DPI = 800, 100
figsize_in = figsize_px / DPI
fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=DPI)
spec = gridspec.GridSpec(ncols=grid_x, nrows=grid_y, figure=fig)

zz = grid_resolution[2] // 2 # middle slice
for yy in range(grid_y):
    for xx in range(grid_x):
        #coeffs = sh_coeffs[:,0,yy,xx,:]
        coeffs = sh_coeffs[xx,yy,zz,:,:]
        #f = calculate_total_sh(coeffs, normalized=True)

        #THIS IS FOR HSV COLOR SPACE
        #h = calculate_total_sh(coeffs[0:1,], normalized=True)
        #s = calculate_total_sh(coeffs[1:2:,], normalized=True)
        #v = calculate_total_sh(coeffs[2:3,], normalized=True)
        #have to map back into rgb
        #hsv = np.stack((h, s, v), axis=-1)
        #rgb = hsv_to_rgb(hsv)

        #THIS IS FOR RGB COLOR SPACE
        r = calculate_total_sh(coeffs[0:1,], normalized=True)
        g = calculate_total_sh(coeffs[1:2:,], normalized=True)
        b = calculate_total_sh(coeffs[2:3,], normalized=True)
        rgb = np.stack((r, g, b), axis=-1)

        rgb = rgb / np.max(rgb)
        #rgb *= 0.5

        ax = fig.add_subplot(spec[xx, yy], projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
        # Hide axes
        ax.set_axis_off()

plt.tight_layout()
plt.savefig('probegrid.png')
plt.show()
