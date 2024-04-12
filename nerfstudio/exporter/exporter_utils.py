# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export utils such as structs, point cloud generation, and rendering code.
"""


from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pymeshlab
import torch
from jaxtyping import Float
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeRemainingColumn)
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn

if TYPE_CHECKING:
    # Importing open3d can take ~1 second, so only do it below if we actually
    # need it.
    import open3d as o3d


@dataclass
class Mesh:
    """Class for a mesh."""

    vertices: Float[Tensor, "num_verts 3"]
    """Vertices of the mesh."""
    faces: Float[Tensor, "num_faces 3"]
    """Faces of the mesh."""
    normals: Float[Tensor, "num_verts 3"]
    """Normals of the mesh."""
    colors: Optional[Float[Tensor, "num_verts 3"]] = None
    """Colors of the mesh."""


def get_mesh_from_pymeshlab_mesh(mesh: pymeshlab.Mesh) -> Mesh:  # type: ignore
    """Get a Mesh from a pymeshlab mesh.
    See https://pymeshlab.readthedocs.io/en/0.1.5/classes/mesh.html for details.
    """
    return Mesh(
        vertices=torch.from_numpy(mesh.vertex_matrix()).float(),
        faces=torch.from_numpy(mesh.face_matrix()).long(),
        normals=torch.from_numpy(np.copy(mesh.vertex_normal_matrix())).float(),
        colors=torch.from_numpy(mesh.vertex_color_matrix()).float(),
    )


def get_mesh_from_filename(filename: str, target_num_faces: Optional[int] = None) -> Mesh:
    """Get a Mesh from a filename."""
    ms = pymeshlab.MeshSet()  # type: ignore
    ms.load_new_mesh(filename)
    if target_num_faces is not None:
        CONSOLE.print("Running meshing decimation with quadric edge collapse")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces)
    mesh = ms.current_mesh()
    return get_mesh_from_pymeshlab_mesh(mesh)

def generate_spherical_harmonics_grid(
    pipeline: Pipeline,
    num_directions: int = 1024,
    grid_res: int = 16,
    rgb_output_name: str = "rgb",
    use_bounding_box: bool = True,
    bounding_box_min: Optional[Tuple[float, float, float]] = None,
    bounding_box_max: Optional[Tuple[float, float, float]] = None,
    crop_obb: Optional[OrientedBox] = None,
) -> o3d.geometry.PointCloud:
    """Generate a spherical harmonics cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        rgb_output_name: Name of the RGB output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.

    Returns:
        GRID of SH Coeffs.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Spherical Harmonics Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    rgbs = []
    view_directions = []
    results = []
    linepoints = torch.linspace(-1, 1, grid_res, device=pipeline.device)
    x,y,z = torch.meshgrid(linepoints, linepoints, linepoints, indexing='ij')
    positions = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    num_rays = positions.shape[0]
    #Samples have frustums which store the directions.  So you should only sample
    #each ray once with the desired direction
    #the rays will be of zero direction
    #the origins will be the sample location because the rays are zero length
    pixel_areas = torch.ones((num_rays, 1), device=pipeline.device)
    nears = torch.zeros((num_rays, 1), device=pipeline.device)
    fars = torch.zeros((num_rays, 1), device=pipeline.device)
    camera_indices = torch.zeros((num_rays, 1), dtype=torch.long, device=pipeline.device)

    with progress as progress_bar:
        task = progress_bar.add_task("Computing Spherical Harmonics Cloud", total=num_directions)
        for i in range(num_directions):
            #random uniform directions
            random_vector = torch.rand((1,3), device=pipeline.device) * 2.0 - 1.0  #move from [0,1] to [-1, 1] 
            rand_direction = random_vector / torch.linalg.norm(random_vector, dim=1, keepdim=True)

            view_directions = torch.ones((num_rays, 3), device=pipeline.device)
            view_directions = view_directions * rand_direction


            bundle = RayBundle(
                origins = positions, directions = view_directions, pixel_area = pixel_areas, nears=nears, fars=fars, camera_indices=camera_indices
            )
            with torch.no_grad():
                outputs = pipeline.model(bundle)

            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            rgbs = rgba[..., :3]
            results.append(rgbs)
            progress.advance(task, 1)
    rgbs_mean = torch.stack(results, dim=0).mean(dim=0)
        
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs_mean.double().cpu().numpy())

    CONSOLE.print("[bold green]:white_check_mark: Done Computing Spherical Harmonics Cloud")
    return pcd

def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    reorient_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = True,
    bounding_box_min: Optional[Tuple[float, float, float]] = None,
    bounding_box_max: Optional[Tuple[float, float, float]] = None,
    crop_obb: Optional[OrientedBox] = None,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        reorient_normals: Whether to re-orient the normals based on the view direction.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    points = []
    rgbs = []
    normals = []
    view_directions = []
    if use_bounding_box and (crop_obb is not None and bounding_box_max is not None):
        CONSOLE.print("Provided aabb and crop_obb at the same time, using only the obb", style="bold yellow")
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            normal = None

            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                assert isinstance(ray_bundle, RayBundle)
                outputs = pipeline.model(ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            depth = outputs[depth_output_name]
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.5
            mask = rgba[..., -1] > 0.5
            point = point[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if use_bounding_box:
                if crop_obb is None:
                    comp_l = torch.tensor(bounding_box_min, device=point.device)
                    comp_m = torch.tensor(bounding_box_max, device=point.device)
                    assert torch.all(
                        comp_l < comp_m
                    ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                    mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
                else:
                    mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            view_directions.append(view_direction)
            if normal is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    view_directions = torch.cat(view_directions, dim=0).cpu()

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
        if ind is not None:
            view_directions = view_directions[ind]

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    # re-orient the normals
    if reorient_normals:
        normals = torch.from_numpy(np.array(pcd.normals)).float()
        mask = torch.sum(view_directions * normals, dim=-1) > 0
        normals[mask] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd


def render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rgb_output_name: str,
    depth_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    disable_distortion: bool = False,
    return_rgba_images: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        disable_distortion: Whether to disable distortion.
        return_rgba_images: Whether to return RGBA images (default RGB).

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":cloud: Computing rgb and depth images :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, disable_distortion=disable_distortion
            ).to(pipeline.device)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if return_rgba_images:
                image = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            else:
                image = outputs[rgb_output_name]
            images.append(image.cpu().numpy())
            depths.append(outputs[depth_output_name].cpu().numpy())
    return images, depths


def collect_camera_poses_for_dataset(dataset: Optional[InputDataset]) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        transform = cameras.camera_to_worlds[idx].tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames


def collect_camera_poses(pipeline: VanillaPipeline) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collects camera poses for train and eval datasets.

    Args:
        pipeline: Pipeline to evaluate with.

    Returns:
        List of train camera poses, list of eval camera poses.
    """

    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    train_frames = collect_camera_poses_for_dataset(train_dataset)
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames


def sample_sphere_low_discrepancy(device : torch.device, num_samples : int = 4096):
    #Capitulum Sampling
    import math
    '''
    CapPoint[I_, maxI_] := Block[{i = I, maxi = maxI}, 
        theta = ArcCos[1 - i / maxI];
        phi = \[Phi]*i;
        {Cos[phi]*Sin[theta], Sin[phi]*Sin[theta], Cos[theta]}];
    samples = Map[CapPoint[#, 500] &, Range[1, 500]];
    samples = Flatten[{samples, -samples}, 1];
    ListPointPlot3D[samples, BoxRatios -> {1, 1, 1.}]
    '''
    Phi = 1.6180339887
    # YES THIS CAN BE VECTORIZED... FIRST LETS MAKE SURE IT IS CORRECT!!!!!
    samples = torch.zeros(num_samples, 3, device = device)
    for i in range(1,num_samples + 1):
        ii = i -1
        theta = math.acos(1 - i / num_samples)
        phi = Phi*i
        samples[ii, 0] = math.cos(phi)*math.sin(theta)
        samples[ii, 1] = math.sin(phi)*math.sin(theta)
        samples[ii, 2] = math.cos(theta)
    samples = torch.cat((samples, samples*-1), 0)
    return samples


def sample_sphere_monte_carlo(device : torch.device, num_samples : int = 4096):
    '''
    xi = RandomReal[{0, 1}, {1000, 3}];
    samples = Map[Block[{x = #},
        a = x[[1]] * 2 - 1;
        b = x[[2]]* 2*\[Pi];
        sinTheta = Sqrt[1 - a*a];
        {Cos[b]*sinTheta, a, Sin[b]*sinTheta}] &, xi];
    ListPointPlot3D[samples, BoxRatios -> {1, 1, 1}]
    '''
    import math
    xi = torch.rand((num_samples, 3), device = device)
    samples = torch.zeros(num_samples, 3, device = device)
    for i in range(num_samples):
        x = xi[i]
        a = x[0] * 2 - 1
        b = x[1] * 2 * math.pi
        sinTheta = math.sqrt(1 - a*a)
        samples[i, 0] = math.cos(b) * sinTheta
        samples[i, 1] = a
        samples[i, 2] = math.sin(b) * sinTheta
    return samples

def project_sh(L : int , M : int, n : torch.FloatTensor):
    '''
    __host__ __device__ __forceinline__ float Project(const vec3& n, const int L, const int M)
    {
        switch (L)
        {
        case 0:
            return 0.2820947917738781f;
        case 1:
        {
            switch (M)
            {
            case -1:	return 0.4886025119029199f * n.y;
            case 0:		return 0.4886025119029199f * n.z;
            case 1:		return 0.4886025119029199f * n.x;
            }
        }
        case 2:
        {
            switch (M)
            {
            case -2:	return 1.0925484305920792f * n.x * n.y;
            case -1:	return 1.0925484305920792f * n.y * n.z;
            case 0:		return 0.3153915652525200f * (-sqr(n.x) - sqr(n.y) + 2 * sqr(n.z));
            case 1:		return 1.0925484305920792f * n.z * n.x;
            case 2:		return 0.5462742152960396f * (sqr(n.x) - sqr(n.y));
            }
        }
        case 3:
        {
            switch (M)
            {
            case -3: return 0.5900435899266435f * (3 * sqr(n.x) - sqr(n.y)) * n.y;
            case -2: return 2.890611442640554f * n.x * n.y * n.z;
            case -1: return 0.4570457994644658f * n.y * (4 * sqr(n.z) - sqr(n.x) - sqr(n.y));
            case 0: return 0.3731763325901154f * n.z * (2 * sqr(n.z) - 3 * sqr(n.x) - 3 * sqr(n.y));
            case 1: return 0.4570457994644658f * n.x * (4 * sqr(n.z) - sqr(n.x) - sqr(n.y));
            case 2: return 1.445305721320277f * (sqr(n.x) - sqr(n.y)) * n.z;
            case 3: return 0.5900435899266435f * (sqr(n.x) - 4 * sqr(n.y)) * n.x;
            }
        }
        }
        printf("Invalid SH index L = %i, M = %i\n", L, M);
        CudaAssert(false);
    }

    '''
    def sqr(x):
        return x*x
    ix = 0; iy = 1; iz = 2
    if(L==0):
        return 0.2820947917738781
    elif(L==1):
        if(M==-1):	return 0.4886025119029199 * n[:,iy]
        elif(M==0):	return 0.4886025119029199 * n[:,iz]
        elif(M==1):	return 0.4886025119029199 * n[:,ix]
    elif(L==2):
        if(M==-2):	return 1.0925484305920792 * n[:,ix] * n[:,iy]
        elif(M==-1):return 1.0925484305920792 * n[:,iy] * n[:,iz]
        elif(M==0):	return 0.3153915652525200 * (-sqr(n[:,ix]) - sqr(n[:,iy]) + 2 * sqr(n[:,iz]))
        elif(M==1):	return 1.0925484305920792 * n[:,iz] *n[:,ix] 
        elif(M==2):	return 0.5462742152960396 * (sqr(n[:,ix]) - sqr(n[:,iy]))
    elif(L==3):
        if(M==-3): return 0.5900435899266435 * (3 * sqr(n[:,ix]) - sqr(n[:,iy])) * n[:,iy]
        elif(M==-2): return 2.890611442640554 * n[:,ix] * n[:,iy] * n[:,iz]
        elif(M==-1): return 0.4570457994644658 * n[:,iy] * (4 * sqr(n[:,iz]) - sqr(n[:,ix]) - sqr(n[:,iy]))
        elif(M==0): return 0.3731763325901154 * n[:,iz] * (2 * sqr(n[:,iz]) - 3 * sqr(n[:,ix]) - 3 * sqr(n[:,iy]))
        elif(M==1): return 0.4570457994644658 * n[:,ix] * (4 * sqr(n[:,iz]) - sqr(n[:,ix]) - sqr(n[:,iy]))
        elif(M==2): return 1.445305721320277 * (sqr(n[:,ix]) - sqr(n[:,iy])) * n[:,iz]
        elif(M==3): return 0.5900435899266435 * (sqr(n[:,ix]) - 4 * sqr(n[:,iy])) * n[:,ix]
    print("Invalid SH index L = %i, M = %i\n", L, M)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    samples = sample_sphere_low_discrepancy(torch.device('cpu'), 1000)
    #samples = sample_sphere_monte_carlo(torch.device('cpu'), 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[:,0], samples[:,1], samples[:,2])
    plt.show()