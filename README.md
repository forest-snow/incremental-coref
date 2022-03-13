# Adapting Coreference Resolution Models through Active Learning

The code in this repository is associated with "Adapting Coreference Resolution Models through Active Learning" (Yuan et al., ACL 2022). It is forked from [this repository](https://github.com/pitrack/incremental-coref), which is associated with 
[Moving on from OntoNotes: Coreference Resolution Model Transfer](https://arxiv.org/abs/2104.08457) (Xia and Van Durme, EMNLP 2021). 

The active learning simulations make use of the ICoref model. If you are just looking to use the ICoref model without active learning, it is recommended to go to Patrick's repo linked above.

## Active Learning Simulations

To reproduce and run active learning simulations, please look in the `simulation` directory. The README inside the directory will instruct you how to run scripts within the directory.

## User Interface 

To use our UI for labeling coreference, please look in the `ui` directory. The README provides instructions for running the interface locally on a browser.

## Contact

Please contact Michelle Yuan at myuan@cs.umd.edu for any questions or concerns. 


## Citing our work

If you use this repo or the ICoref model, please cite:

```
@inproceedings{yuan-2022,
  title={Adapting Active Learning Models through Active Learning},
  author={Michelle Yuan and Patrick Xia and Chandler May and Benjamin {Van Durme} and Jordan Boyd-Graber},
  year={2022},
  booktitle={Proceedings of EMNLP},
}
```

```
@inproceedings{xia-etal-2020-incremental,
    title = "Incremental Neural Coreference Resolution in Constant Memory",
    author = "Xia, Patrick  and
      Sedoc, Jo{\~a}o  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    url = "https://aclanthology.org/2020.emnlp-main.695",
    doi = "10.18653/v1/2020.emnlp-main.695",
}
```

```
@inproceedings{xia-van-durme-2021-moving,
  title={Moving on from OntoNotes: Coreference Resolution Model Transfer},
  author={Patrick Xia and Benjamin {Van Durme}},
  year={2021},
  booktitle={Proceedings of EMNLP},
}
```


