# Active Learning Simulations 


## Getting started

To use the code, the high-level process is to convert data into a minimized jsonlines format and create a [jsonnet](https://jsonnet.org/) experiment configuration. Then, running `python active.py <experiment name>` should run the active learning simulation.

### Local setup

In this directory, create `local.jsonnet`, which is the local configuration file. It should contain the following information

```
{
  data_dir: "", // location of data files
  encoders_dir: "", // location of encoders, could be the same as above
  log_root: "", // location of logs
  gpu_gb: "", // max gpu limit, in GB. Used in gradient accumulation
  simulation_dir: "" // location of simulation output files
}
```

For packages, the primary ones are jsonnet(=0.16.0), torch(=1.7.0), and transformers(=3.4.0), and these are also included in requirements.txt. The repo is not tested on the most recent versions of these libraries. 

To set up all of this, run `pip install -r requirements.txt`.

### Source model

In the paper, we use the best checkpoint of ICoref trained on OntoNotes (from EMNLP 2020) as the source model to initialize the active learning simulation. Please download the [`checkpoint.bin` here](https://nlp.jhu.edu/incremental-coref/models/checkpoint.bin) (1.7GB)
and place it under this directory.

### Defining experiments

The config files are sorted into [jsonnets](https://jsonnet.org/), which is a data templating language. The main model parameters are defined in `jsonnets/base.jsonnet`, encoder parameters in `jsonnets/encoder.jsonnet`, and data parameters in `jsonnets/data.jsonnet`. Local paths, as mentioned above, are in `local.jsonnet`. The remaining files are individual files related to different experiments/projects. `jsonnets/verify_jsonnet.py` wraps the jsonnet import and load function to quickly check for syntax errors by running `python verify_jsonnet.py <jsonnet_file>`.

The experiments themselves are then imported at the top-level `experiments.jsonnet`. This is the file ultimately read by the code.


### Datasets 

**OntoNotes**

Download OntoNotes from [OntoNotes 5.0, official LDC release](https://catalog.ldc.upenn.edu/LDC2013T19). Run `conversion_scripts/minimize.py` (same as in [prior work](https://github.com/mandarjoshi90/coref)) to convert the OntoNotes format to jsonlines. Place these into `$data_dir/ontonotes`. 

**PreCo**

Download PreCo. If you are having trouble finding the dataset, you can email Patrick Xia for help. Then, run

```map_preco.py {trainl, dev}.json {train, dev}.jsonlines```

over the two files to create jsonlines, which can then be further preprocessed into segments of size 512 with `minimize_json.py`.

**QBCoref**

Download [QBCoref](https://www.anupamguha.com/qbcoreference). If you are having trouble finding the dataset, you can also email Patrick Xia for help. Next, run

``python conversion_scripts/minimize_qb.py data-gold data-gold-json bert``

and make the splits with

``python conversion_scripts/make_qbcoref_splits.py all_docs.512.jsonlines``.



## Running simulations

### Reproducing experiments in paper

To reproduce the experiments in the paper, you simply have to modify and run
```
bash simulation.sh
```

In `simulation.sh`, you can change:
* SEED: random seed initialization
* STRATEGY: the active learning sampling strategy
* NUM_SPANS: number of spans sampled each cycle
* MAX_DOCS: maximum number of docs read (spans sampled must belong to no more than this number)
* NUM_CYCLES: number of active learning cycles
* DATASET: the target dataset to transfer to

The script also comments possible values for the above parameters. The results of the experiments will be placed in $simulation_dir.


### Extending experiments

If you'd like to extend experiments beyond what's described in the paper, you will need to modify `jsonnets/active.jsonnet`. To include more parameter configurations for PreCo, extend the values under `active_preco`. The same can be done for QBCoref under `active_qbcoref`.

To modify the active learning pipeline (e.g. develop more strategies), you can look into `active.py`, which is the main file for running simulations, and `score.py`, which scores spans based on a certain criteria like joint entropy. 

