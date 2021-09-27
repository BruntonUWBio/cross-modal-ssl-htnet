#!/bin/bash
# Test cross-modal deep clustering on finger flexion dataset
# (modified from from https://github.com/yukimasano/self-label)
device="0"
DIR='.../ecog_finger_flexion_xarray/ .../ecog_finger_flexion_xarray/pose/'
subjects='S01 S02 S03 S04 S05 S07 S08 S09'
run_num=1
savepath=.../xdc_runs/fingerflex
param_lp=../models/fingerflex  # model parameter file to use

# the network
ARCH='htnet htnet'
USE_ECOG='True True' # both True b/c pose srate = EEG srate
HC=1
NCL=5 # number of expected clusters (K)
NFOLDS="10" # number of folds (can also be "loocv")

# the training
WORKERS=1 # number of workers
BS=36 # batch size
AUG=0

#########CROSS-MODAL MODEL#########

IS_SUP="False" # is supervised
IS_XDC="True" # is cross-modal
NEP=200 # number of epochs
nopts=50 # number of times to update pseudo-labels

for SBJ in $subjects
do
    python3 main_n_multimodel_cv.py \
            --device ${device} \
            --imagenet-path ${DIR} \
            --batch-size ${BS} \
            --augs ${AUG} \
            --epochs ${NEP} \
            --nopts ${nopts} \
            --hc ${HC} \
            --arch ${ARCH} \
            --ncl ${NCL} \
            --workers ${WORKERS} \
            --pat_id ${SBJ} \
            --use_ecog ${USE_ECOG} \
            --n_folds ${NFOLDS} \
            --is_supervised ${IS_SUP} \
            --is_xdc ${IS_XDC} \
            --savepath ${savepath}_xdc_run${run_num}/ \
            --param_lp ${param_lp};
done


#########SUPERVISED MODEL#########

IS_SUP="True" # is supervised
IS_XDC="False" # is cross-modal
NEP=40 # number of epochs
nopts=20 # number of times to update pseudo-labels

for SBJ in $subjects
do
    python3 main_n_multimodel_cv.py \
            --device ${device} \
            --imagenet-path ${DIR} \
            --batch-size ${BS} \
            --augs ${AUG} \
            --epochs ${NEP} \
            --nopts ${nopts} \
            --hc ${HC} \
            --arch ${ARCH} \
            --ncl ${NCL} \
            --workers ${WORKERS} \
            --pat_id ${SBJ} \
            --use_ecog ${USE_ECOG} \
            --n_folds ${NFOLDS} \
            --is_supervised ${IS_SUP} \
            --is_xdc ${IS_XDC} \
            --savepath ${savepath}_supervised_run${run_num}/ \
            --param_lp ${param_lp};
done


#########UNIMODAL MODEL#########

IS_SUP="False" # is supervised
IS_XDC="False" # is cross-modal
NEP=40 # number of epochs
nopts=20 # number of times to update pseudo-labels

for SBJ in $subjects
do
    python3 main_n_multimodel_cv.py \
            --device ${device} \
            --imagenet-path ${DIR} \
            --batch-size ${BS} \
            --augs ${AUG} \
            --epochs ${NEP} \
            --nopts ${nopts} \
            --hc ${HC} \
            --arch ${ARCH} \
            --ncl ${NCL} \
            --workers ${WORKERS} \
            --pat_id ${SBJ} \
            --use_ecog ${USE_ECOG} \
            --n_folds ${NFOLDS} \
            --is_supervised ${IS_SUP} \
            --is_xdc ${IS_XDC} \
            --savepath ${savepath}_sep_run${run_num}/ \
            --param_lp ${param_lp};
done