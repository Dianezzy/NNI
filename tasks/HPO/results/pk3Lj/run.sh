#!/bin/bash
cd '/home/baidonglin/Projects/nni/nas/.'
export NNI_PLATFORM='local'
export NNI_EXP_ID='XUxjyZuG'
export NNI_SYS_DIR='/home/baidonglin/nni/experiments/XUxjyZuG/trials/pk3Lj'
export NNI_TRIAL_JOB_ID='pk3Lj'
export NNI_OUTPUT_DIR='/home/baidonglin/nni/experiments/XUxjyZuG/trials/pk3Lj'
export NNI_TRIAL_SEQ_ID='0'
export MULTI_PHASE='false'
export CUDA_VISIBLE_DEVICES='0'
eval python main_nas.py --epochs 300 --num_workers 8 2>"/home/baidonglin/nni/experiments/XUxjyZuG/trials/pk3Lj/stderr"
echo $? `date +%s%3N` >'/home/baidonglin/nni/experiments/XUxjyZuG/trials/pk3Lj/.nni/state'