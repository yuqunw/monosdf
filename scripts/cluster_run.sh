#!/bin/bash
run_name="monosdf_eth3d_smaller_bias"
echo "Args provided: $1 $2"
scene=$1
if [[ $2 == "--with_full" ]]; then
    with_full=True
    run_name+="_full"
else
    with_full=False
fi

echo "Running with run name ${run_name}"
MOUNT_DIR="/home/yuqunwu2/large_scale_nerf/data/$(hostname -s)"
DATA_DIR="${MOUNT_DIR}/eth3d_processed_monosdf"
GT_DIR="${MOUNT_DIR}/eth3d_ground_truths"
CHECKPOINT_DIR="${MOUNT_DIR}/${run_name}"
mkdir -p $CHECKPOINT_DIR
rm -r "${CHECKPOINT_DIR}/${scene}"

cd /home/yuqunwu2/large_scale_nerf/monosdf/code

python training/exp_runner.py --scan_id ${scene} \
                              --full ${with_full} \
                              --expname ${scene} \
                              --data_root ${MOUNT_DIR} \
                              --exps_folder ${run_name}


python evaluation/generate_img_mesh.py --checkpoint "${MOUNT_DIR}/${run_name}/${scene}/checkpoints/ModelParameters/latest.pth" \
                                       --evals_folder "${MOUNT_DIR}/${run_name}/${scene}/output" \
                                       --scan_id ${scene} \
                                       --data_root ${MOUNT_DIR}

# python evaluation/evaluate_single_scene.py --input_path "${DATA_DIR}/${scene}" \
#                          --output_path "${CHECKPOINT_DIR}/${scene}/output" \
#                          --gt_path "${GT_DIR}/${scene}" 

cd ${CHECKPOINT_DIR}/${scene}/
zip -r "${scene}.zip" "output"

result_dirname="ablations"
ssh jae "mkdir -p /mnt/data1/cluster_results/${result_dirname}/${run_name}"
ssh jae "mkdir -p /mnt/data1/eth3d_outputs/${result_dirname}/${run_name}/${scene}"
scp "${scene}.zip" jae:/mnt/data1/cluster_results/${result_dirname}/${run_name}/
ssh jae "cd /mnt/data1/cluster_results/${result_dirname}/${run_name}/ && unzip ${scene}.zip -d /mnt/data1/eth3d_outputs/${result_dirname}/${run_name}/${scene}"

