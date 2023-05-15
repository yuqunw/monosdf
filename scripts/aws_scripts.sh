#!/bin/bash
run_name="monosdf_eth3d"
echo "Args provided: $1 $2 $3"
scene=$1
export CUDA_VISIBLE_DEVICES=$3
if [[ $2 == "--with_full" ]]; then
    with_full=True
    run_name+="_full"
else
    with_full=False
fi

echo "Running with run name ${run_name}"
MOUNT_DIR="/home/ubuntu"
DATA_DIR="${MOUNT_DIR}/eth3d_processed_monosdf"
CHECKPOINT_DIR="${MOUNT_DIR}/${run_name}"
mkdir -p $CHECKPOINT_DIR
rm -r "${CHECKPOINT_DIR}/${scene}"

cd /home/ubuntu/monosdf/code

python training/exp_runner.py --scan_id ${scene} \
                              --full ${with_full} \
                              --expname ${scene} \
                              --data_root ${MOUNT_DIR} \
                              --exps_folder ${run_name}


python evaluation/generate_img_mesh.py --checkpoint "${MOUNT_DIR}/${run_name}/${scene}/checkpoints/ModelParameters/latest.pth" \
                                       --evals_folder "${MOUNT_DIR}/${run_name}/${scene}/output" \
                                       --scan_id ${scene} \
                                       --data_root ${MOUNT_DIR}

cd ${CHECKPOINT_DIR}/${scene}/
zip -r "${scene}.zip" "output"
aws s3 cp "${scene}.zip" s3://uiuc-jae-data/monosdf_run/${run_name}/${scene}.zip
