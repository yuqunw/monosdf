import os
import sys
sys.path.append('.')
sys.path.append('..')

from pathlib import Path
import json
from PIL import Image
from lpips import LPIPS
from pytorch_msssim import ssim
import torchvision.transforms.functional as F
import torch
import numpy as np
from tqdm import tqdm
import math

eth3d_evaluation_bin = Path('/home/jae/dev/multi-view-evaluation/build/ETH3DMultiViewEvaluation')
gt_files_path = Path('/mnt/data1/eth3d_ground_truths')
processed_files_path = Path('/mnt/data1/eth3d_processed')

scenes = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']

def evaluate_3d(result_path, gt_path):
    rec_file = result_path / 'fused.ply'
    gt_file = gt_path / "dslr_scan_eval" / "scan_alignment.mlp"
    exe_str = f'{str(eth3d_evaluation_bin)} --reconstruction_ply_path {rec_file} --ground_truth_mlp_path {gt_file} --tolerances 0.02,0.05'
    output = os.popen(exe_str).read()
    print(output)
    lines = output.split('\n')
    tolerances = [0.02, 0.05]
    com_index = [i for i, line in enumerate(lines) if line.find('Completeness') == 0][0]
    com_line = lines[com_index]
    acc_line = lines[com_index+1]
    f1_line = lines[com_index+2]
    print(com_line)
    print(acc_line)
    print(f1_line)
    com_words = com_line.split()
    acc_words = acc_line.split()
    f1_words = f1_line.split()
    ress = {}
    for i, tol in enumerate(tolerances):
        res ={}
        res[f'completeness'] = float(com_words[i + 1])
        res[f'accuracy'] = float(acc_words[i + 1])
        res[f'f1'] = float(f1_words[i + 1])
        ress[f'tol_{tol}'] = res

    return ress

def measure_psnr(ref_image, src_image, mask = None):
    mse = ((ref_image - src_image) ** 2).mean()
    return -10.0 * math.log10(mse.item())

lpips_fn = LPIPS(net='alex').to('cuda').eval()

def measure_lpips(ref_image, src_image, mask = None):
    with torch.no_grad():
        return lpips_fn(ref_image[None].cuda(), src_image[None].cuda(), normalize=True).cpu().item()

def measure_ssim(ref_image, src_image, mask = None):
    return ssim(ref_image[None], src_image[None], data_range=1.0).item()

def evaluate_images(result_path, input_path):
    with open(input_path / 'transforms_test.json') as f:
        trans = json.load(f)
    frames = trans['frames']
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    for frame in tqdm(frames, desc='Evaluating images...', leave=False, dynamic_ncols=True):
        gt_file_path= input_path / frame['file_path']
        inf_file_path = result_path / frame['file_path']
        
        mask_dir = Path('/mnt/data/eth3d_processed')
        mask_path = mask_dir / ('dynamic_mask_' + frame['file_path'].split('/')[-1])
        with Image.open(mask_path) as img:
            mask = 1 - F.to_tensor(img)
 
        gt_image = Image.open(gt_file_path)
        inf_image = Image.open(inf_file_path)

        gt_rgb = F.to_tensor(gt_image)[mask]
        inf_rgb = F.to_tensor(inf_image)[mask]

        # compute PSNR, SSIM, LPIPS
        psnr_val = measure_psnr(inf_rgb, gt_rgb)
        ssim_val = measure_ssim(inf_rgb, gt_rgb)
        lpips_val = measure_lpips(inf_rgb, gt_rgb)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

    # compute average and return
    return {
        'psnr': np.array(psnr_vals).mean(),
        'ssim': np.array(ssim_vals).mean(),
        'lpips': np.array(lpips_vals).mean(),
    }

def main(args):
    scene = args.scene

    # first perform fusion
    input_path = processed_files_path / scene
    gt_path = gt_files_path / scene
    Path(args.eval_path).parent.mkdir(exist_ok=True, parents=True)
    result_path = Path(args.result_path) / scene
    full_result_path = Path(args.result_path + '_full') / scene


    # if not (result_path / 'fused.ply').exists():
    #     fuse_reconstruction(str(full_result_path), args.threshold, args.min_views, args.device)

    # evaluate images for test samples
    evals_images = evaluate_images(result_path, input_path)

    # evaluate 3d 
    # evals_3d = evaluate_3d(full_result_path, gt_path)
    evals_3d = {}

    # write evaluation results to file
    evals = {**evals_images, **evals_3d}
    with open(Path(args.eval_path), 'w') as f:
        json.dump(evals, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--scene', default='office', type=str, choices=scenes)
    parser.add_argument('--threshold', default=2.0, type=float)
    parser.add_argument('--min_views', default=2, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    main(args)
