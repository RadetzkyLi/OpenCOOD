# export necessary environmental viriables
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=0
export MASTER_ADDR='localhost'
export MASTER_PORT='3378'
# fusion strategy
fusion="late"
num_gpus=3
# for continue training
model_dir="/home/lrs/scripts/OpenCOOD/opencood/logs/point_pillar_early_fusion_2023_12_18_16_14_10"  
if [ ${fusion} = "early" ] ; then
    hypes_yaml='/home/lrs/scripts/OpenCOOD/opencood/hypes_yaml/point_pillar_early_fusion.yaml'
elif [ ${fusion} = "intermediate" ] ; then
    hypes_yaml="/home/lrs/scripts/OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion_v2.yaml"
    #hypes_yaml="/home/lrs/scripts/OpenCOOD/opencood/hypes_yaml/point_pillar_v2xvit_v2.yaml"
elif [ ${fusion} = "late" ] ; then
    hypes_yaml="/home/lrs/scripts/OpenCOOD/opencood/hypes_yaml/point_pillar_late_fusion_v2.yaml"
else
    echo "Unexpected fusion method: ${fusion}"
    exit 1
fi 
# start training
echo "Fusion method: ${fusion}"
# python /home/lrs/scripts/OpenCOOD/opencood/tools/train.py --hypes_yaml ${hypes_yaml} 
python -m torch.distributed.launch --nproc_per_node ${num_gpus} --use_env /home/lrs/scripts/OpenCOOD/opencood/tools/train.py\
 --hypes_yaml ${hypes_yaml}
