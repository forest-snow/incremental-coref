#!/bin/bash
set -e

### PARAMETERS ###

# [67, 312, 57, 29, 8]
SEED=67
# ["random", "ment-ent", "clust-ent", "cond-ent", "joint-ent", "noise", li-ent"]
STRATEGY='joint-ent'
# preco: [0, 20, 50], qbcoref: [0, 20, 40]
NUM_SPANS=50
# preco: [0, 1, 5, 20, 50], qbcoref: [0, 1, 5, 20, 40]
MAX_DOCS=1
# preco: [0, 6, 15], qbcoref: [0, 10, 20]
NUM_CYCLES=6
# ["preco", "qbcoref"]
DATASET='preco'

######

# Debugging experiments
python active.py active_debug

# Run simulation
#python active.py "active_${DATASET}_${SEED}_${NUM_SPANS}_${MAX_DOCS}_${STRATEGY}_${NUM_CYCLES}"

