#!/bin/bash
# Script for running all bash scripts at once
./scripts/ecog_move_rest.sh
./scripts/eeg_move_rest.sh
./scripts/ecog_fingerflex.sh
./scripts/eeg_balance_perturbations.sh
./scripts/htnet_fingerflex_cluster.sh