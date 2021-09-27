#!/bin/bash
# Test cross-modal deep clustering on ECoG move/rest dataset
# (modified from from https://github.com/yukimasano/self-label)
device="0"
DIR='.../ecog_move_rest/ .../ecog_move_rest/pose/'
subjects='EC01 EC02 EC03 EC04 EC05 EC06 EC07 EC08 EC09 EC10 EC11 EC12'
run_num=1
savepath=.../xdc_runs/naturalistic
param_lp=../models/naturalistic  # model parameter file to use

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