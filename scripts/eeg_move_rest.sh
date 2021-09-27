#!/bin/bash
# Test cross-modal deep clustering on EEG move/rest dataset
# (modified from from https://github.com/yukimasano/self-label)
device="0"
DIR='.../eeg_move_rest_xarray/ .../eeg_move_rest_xarray/pose/'
subjects='S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12 S13 S14 S15'
run_num=1
savepath=.../xdc_runs/eeg_move_rest
param_lp=../models/eeg_move_rest  # model parameter file to use

# the network
ARCH='htnet htnet'
USE_ECOG='True True' # both True b/c pose srate = EEG srate
HC=1
NCL=2 # number of expected clusters (K)
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