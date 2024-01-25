import sys

sys.path.append('../code')
# sys.path.remove('/home/zhenglinyu2/anaconda3/envs/monosdf/lib/python3.7/site-packages')
import argparse
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
from training.monosdf_train import MonoSDFTrainRunner
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/tnt_highres_mlp.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="monosdf_eth3d")
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=str, default=None, help='If set, taken to be the scan name.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument("--full", type=eval, default=True, choices=[True, False], help='whether use full scene for training')
    parser.add_argument("--data_root", type=str, default='../../data/vision-xx', help='the data root for the cluster:../../data/vision-xx')

    opt = parser.parse_args()

    '''
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    '''
    gpu = opt.local_rank

    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    print(opt.local_rank)
    torch.cuda.set_device(opt.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    # torch.distributed.barrier()
    
    if opt.scan_id in ['Barn', 'Courthouse']:
        opt.conf = './confs/tnt_highres_mlp_outside_in.conf'
    else:
        opt.conf = './confs/tnt_highres_mlp_inside_out.conf'
        
    trainrunner = MonoSDFTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    full=opt.full,
                                    data_root = opt.data_root
                                    )

    trainrunner.run()
