import sys
sys.path.append('../code')
import argparse
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import open3d as o3d
import multiprocessing as mp

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from evaluation.evaluate_single_scene import cull_scan

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def generate_pc_from_mesh(scan_id, mesh_path, pc_path, thresh):
    mp.freeze_support()
    pbar = tqdm(total=9)
    pbar.set_description('read data mesh')
    data_mesh = o3d.io.read_triangle_mesh(mesh_path)

    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    # pbar.update(1)
    # pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:,1] - tri_vert[:,0]
    v2 = tri_vert[:,2] - tri_vert[:,0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:,0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_pcd)
    o3d.io.write_point_cloud(pc_path, point_cloud)

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])

    evals_folder_name = kwargs['evals_folder_name']
    results_folder_name = evals_folder_name + '/results'
    generate_image = kwargs['generate_image']
    generate_mesh = kwargs['generate_mesh']
    thresh = kwargs['thresh']

    scan_id = kwargs['scan_id'] if kwargs['scan_id'] is not None else conf.get_int('dataset.scan_id', default=-1)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] is not None:
        dataset_conf['scan_id'] = kwargs['scan_id']
    
    # use all images for evaluation
    dataset_conf['num_views'] = -1

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()
    saved_model_state = torch.load(str(kwargs['checkpoint']))
    
    # deal with multi-gpu training model
    if list(saved_model_state["model_state_dict"].keys())[0].startswith("module."):
        saved_model_state["model_state_dict"] = {k[7:]: v for k, v in saved_model_state["model_state_dict"].items()}

    model.load_state_dict(saved_model_state["model_state_dict"], strict=True)

    ####################################################################################################################
   
    model.eval()
    if generate_mesh:
        print("Generating mesh...")
        with torch.no_grad():
            grid_boundary=conf.get_list('plot.grid_boundary')
            mesh = plt.get_surface_sliding(path="",
                                        epoch="",
                                        sdf=lambda x: model.implicit_network(x)[:, 0],
                                        resolution=kwargs['resolution'],
                                        grid_boundary=grid_boundary,
                                        level=0,
                                        return_mesh=True
            )

            # Transform to world coordWinates
            # Important!!!!!!!!!!!!!!!!
            # We should not use this, as our scale matrix is actually not stord
            # if kwargs['world_space']:
            #     mesh.apply_transform(scale_mat)

            # Taking the biggest connected component
            #components = mesh.split(only_watertight=False)
            #areas = np.array([c.area for c in components], dtype=np.float32)
            #mesh_clean = components[areas.argmax()]

            utils.mkdir_ifnotexists(results_folder_name)
            mesh_path = '{0}/{1}_mesh.ply'.format(results_folder_name, scan_id)
            mesh.export(mesh_path, 'ply')

        # Cull point cloud from mesh
        # cull_scan(scan, mesh_path, result_mesh_file)
        pc_path = '{0}/{1}.ply'.format(results_folder_name, scan_id)
        # cull_scan(scan_id, mesh_path, pc_path)
        generate_pc_from_mesh(scan_id, mesh_path, pc_path, thresh)

    if generate_image:
        print("Generating image...")
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                collate_fn=eval_dataset.collate_fn
                                                )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res
        split_n_pixels = conf.get_int('train.split_n_pixels', 10000)
    
        images_dir = '{0}/images'.format(evals_folder_name)
        utils.mkdir_ifnotexists(images_dir)

        # psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
            res = []
            for s in tqdm(split):
                torch.cuda.empty_cache()
                out = model(s, indices)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)


            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            # Image.fromarray(np_image).save(output_path / 'images' / f'{i:06d}.rgb.png')
            img.save('{0}/{1}.rgb.png'.format(images_dir,'%06d' % indices[0]))

            # psnr = rend_util.get_psnr(model_outputs['rgb_values'],
            #                           ground_truth['rgb'].cuda().reshape(-1, 3)).item()
            # psnrs.append(psnr)


        # psnrs = np.array(psnrs).astype(np.float64)
        # print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))
        # psnrs = np.concatenate([psnrs, psnrs.mean()[None], psnrs.std()[None]])
        # pd.DataFrame(psnrs).to_csv('{0}/results.csv'.format(results_folder_name))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', type=str, default='./confs/eth3d_highres_mlp.conf')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=str, default='courtyard', help='Scene name')
    parser.add_argument('--resolution', default=1024, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--world_space', default=False, action="store_true", help='If set, transform to world space')
    parser.add_argument('--generate_image', default=True, type = eval, choices=[True, False], help='Whether generate image.')
    parser.add_argument('--generate_mesh', default=True, type = eval, choices=[True, False], help='Whether generate mesh.')
    parser.add_argument('--downsample_density', default=0.1, type=float, help='Downsample density for mesh generation')

    opt = parser.parse_args()

    evaluate(conf=opt.conf,
             evals_folder_name=opt.evals_folder,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             world_space=opt.world_space,
             generate_image=opt.generate_image,
             generate_mesh=opt.generate_mesh,
             thresh = opt.downsample_density
             )
