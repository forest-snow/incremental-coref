import pathlib
import json
import copy
import transformers
import pandas as pd
from analyze_sim import load_data
from metrics import CorefEvaluator
from collections import defaultdict
from eval_all import update_evaluators, ALL
from tqdm import tqdm
import sys

def count_errors_doc(gold, preds, counts):
    entities = {}
    for idx,cluster in enumerate(gold):
        for mention in cluster:
            entities[tuple(mention)] = idx

    pred_gold = []
    covered_entities = set()
    covered_mentions = set()
    for cluster in preds:
        pred_gold_ids= []
        for mention in cluster:
            mention = tuple(mention)
            if mention in entities:
                entity = entities[mention]
                pred_gold_ids.append(entity)
                covered_entities.add(entity)
                covered_mentions.add(mention)
            else:
                pred_gold_ids.append(-1)
        pred_gold.append(pred_gold_ids)

    # check missing entity
    counts['missing_entity'] += len(gold) - len(covered_entities)
    # check missing mentions
    counts['missing_mention'] += len([m for c in gold for m in c]) - len(covered_mentions)
    for cluster in pred_gold:
        if set(cluster) == {-1}:
            # check extra entity
            counts['extra_entity'] += 1

        # check extra mention
        counts['extra_mention'] += sum([1 for x in cluster if x == -1])
        if len(set(cluster)-{-1}) > 1:
            # check conflated
            counts['conflated'] += 1
    for i in range(len(gold)):
        # check divided
        count = 0
        for cluster in pred_gold:
            if count > 1:
                counts['divided'] += 1
                break
            for x in cluster:
                if x == i:
                    count += 1



def count_errors_preds(preds_file):
    counts = {
        'missing_entity':0,
        'extra_entity':0,
        'missing_mention':0,
        'extra_mention':0,
        'divided':0,
        'conflated':0,
            }
    with open(preds_file, 'r') as f:
        for line in tqdm(f):
            doc = json.loads(line)
            count_errors_doc(doc['clusters'], doc['predicted_clusters'],
                    counts)
    print(counts)



if __name__ == "__main__":

    # input should be path to preds.json file from an experiment
    preds_file = pathlib.Path(sys.argv[1])
    count_errors_preds(preds_file)
