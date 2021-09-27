#!/bin/bash
# Test cross-modal deep clustering on EEG perturbations dataset
# (modified from from https://github.com/yukimasano/self-label)
device="0"
DIR='.../eeg_balance_perturbations_xarray/ .../eeg_balance_perturbations_xarray/pose/ .../eeg_balance_perturbations_xarray/emg/'
DIR_eeg_pose='.../eeg_balance_perturbations_xarray/ .../eeg_balance_perturbations_xarray/pose/'
DIR_pose_emg='.../eeg_balance_perturbations_xarray/pose/ .../eeg_balance_perturbations_xarray/emg/'
DIR_eeg_emg='.../eeg_balance_perturbations_xarray/ .../eeg_balance_perturbations_xarray/emg/'
subjects='S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12 S13 S15 S16 S17 S19 S20 S22 S23 S24 S25 S26 S27 S28 S29 S30 S31 S32 S33'
run_num=1
savepath=.../xdc_runs/eeg_balance_perturbations
sp_eeg_pose=.../xdc_runs/eeg_balance_perturbations_eeg_pose
sp_pose_emg=.../xdc_runs/eeg_balance_perturbations_pose_emg
sp_eeg_emg=.../xdc_runs/eeg_balance_perturbations_eeg_emg
param_lp=.../models/eeg_balance_perturbations  # model parameter file to use


#########TRIMODAL MODELS#########

# the network
ARCH='htnet htnet htnet'
USE_ECOG='True True True' # all same srate
HC=1
NCL=4 # number of expected clusters (K)
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



#########PAIRWISE MODELS (all cross-modal)#########

# the network
ARCH='htnet htnet'
USE_ECOG='True True' # all same srate

IS_SUP="False" # is supervised
IS_XDC="True" # is cross-modal
NEP=200 # number of epochs
nopts=50 # number of times to update pseudo-labels


#########EEG-POSE#########

for SBJ in $subjects
do
    python3 main_n_multimodel_cv.py \
            --device ${device} \
            --imagenet-path ${DIR_eeg_pose} \
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
            --savepath ${sp_eeg_pose}_xdc_run${run_num}/ \
            --param_lp ${param_lp};
done


#########POSE-EMG#########

for SBJ in $subjects
do
    python3 main_n_multimodel_cv.py \
            --device ${device} \
            --imagenet-path ${DIR_pose_emg} \
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
            --savepath ${sp_pose_emg}_xdc_run${run_num}/ \
            --param_lp ${param_lp};
done


#########EEG-EMG#########

for SBJ in $subjects
do
    python3 main_n_multimodel_cv.py \
            --device ${device} \
            --imagenet-path ${DIR_eeg_emg} \
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
            --savepath ${sp_eeg_emg}_xdc_run${run_num}/ \
            --param_lp ${param_lp};
done