SEED=67
STRATEGY='joint-ent'
NUM_SPANS=50
MAX_DOCS=1
NUM_CYCLES=6
DATASET='preco'

# Debugging experiments
python active.py active_debug

# Run simulation
#python active.py "active_${DATASET}_${SEED}_${NUM_SPANS}_${MAX_DOCS}_${STRATEGY}_${NUM_CYCLES}"

# PreCo simulations
#python active.py "active_preco_${SEED}_${NUM_SPANS}_${MAX_DOCS}_${STRATEGY}_${NUM_CYCLES}"

# QBCoref simulations
#python active.py "active_qbcoref_${SEED}_${NUM_SPANS}_${MAX_DOCS}_${STRATEGY}_${NUM_CYCLES}"
