import argparse
import torch
import numpy as np
import pathlib
from train_active import finetune_on_queries
from incremental import Incremental
from inference import Predictor
from score import Scorer
import util
import copy
from collections import defaultdict
import logging
import json
from eval_all import ALL


def load_model(config, fp):
    model = Incremental(config)
    util.load_params(model, fp, "model")
    logging.info(f"Updating threshold to {config['threshold']}")
    model.set_threshold(config["threshold"])
    return model


def update_queries(queries, span_scores, n, c, read_docs=None):
    """
    Assumes [span_scores] is sorted in descending order according to score.
    Adds at most [n] entries from [span_scores] to [queries] with highest score.
    Avoids adding duplicate spans.

    While adding queries, constrain added spans from set of at most [c] documents
    by taking top-[c] spans and then constraining the rest of
    added spans to belong only in the same documents as those [c] spans.

    [read_docs] is set containing docs already read (for mixed-ent)
    """
    if read_docs is None:
        read_docs = set()
    num_spans = n

    for span_score in span_scores:
        if n <= 0:
            # stop, already queried enough spans
            break

        doc_key = span_score['doc_key']

        if span_score['span'] in queries.get(doc_key, []):
            # skip duplicate span
            continue

        else:
            if len(read_docs) < c:
                # add span to queries
                queries[doc_key].append(span_score['span'])
                n -= 1
                # add to constrained docs
                read_docs.add(doc_key)

            else:
                if doc_key in read_docs:
                    # only add span to queries if in constrained docs
                    queries[doc_key].append(span_score['span'])
                    n -= 1

    logging.info(f'Added {num_spans-n} spans belonging to {len(read_docs)} docs')
    return read_docs

def sample_spans(model, dataset, queries, config, scoring, cycle):
    """
    Sample spans for active learning by scoring spans based on strategy and
    picking the top-scoring spans. Does not sample duplicate spans that are
    already queried.
    - [model]: acquisition model
    - [dataset]: data to sample from
    - [queries]: spans that are already queried before
    - [config]: simulation hyperparameters
    - [scoring]: active learning strategy
    - [cycle]: the cycle number
    """

    scorer = Scorer(model, dataset)
    SCORING = {
        'random-ment': scorer.random_ment,
        'mention': scorer.mention_detection,
        'cond-ent':scorer.conditional_entropy,
        'random':scorer.random,
        'ment-ent':scorer.mention_entropy,
        'joint-ent':scorer.joint_entropy,
        'clust-ent':scorer.cluster_entropy,
        'li-clust-ent':scorer.li_entropy
        }


    with torch.no_grad():
        span_scores = SCORING[scoring](
                cycle=cycle,
                num_cycles=config['num_cycles']
            )

    # sort scores in descending order
    span_scores.sort(key = lambda k : k['score'], reverse=True)

    # add to [queries]
    update_queries(queries, span_scores, config['num_spans'], config['max_docs'])
    # return all spans with their scores
    return span_scores


def extract_data(dataset, queries):
    """
    From [dataset], extract antecedent annotations for [queries].
    Returns (new_data, num_queries) where:
     - [new_data] is a subset of [dataset] which contains annotations only for spans in [queries]
     - [num_queries] is number of annotated spans in [new_data]
    """
    stop = len(queries)
    new_data = []
    num_queries = 0

    # iterate over [dataset] with stopping condition for faster search
    for doc in dataset:

        if stop <= 0:
            # already queried labels for all sampled spans
            break

        doc_key = doc['doc_key']
        if doc_key in queries:
            # at least one span in doc was queried
            gold_map = doc['antecedent_map']
            # get most recent antecedent for queried spans
            sub_map = {}
            for span in queries.get(doc_key, []):
                if span in gold_map:
                    ante = gold_map[span]
                    if type(ante) is list:
                        # grab most recent antecedent if not '0' cluster
                        ante = [max(ante)]
                    sub_map[span] = ante
                else:
                    sub_map[span] = None
                num_queries += 1

            if len(sub_map) > 0:
                # create new copy of doc with new sub_map
                new_doc = copy.copy(doc)
                new_doc['antecedent_map'] = sub_map
                new_data.append(new_doc)
            stop -= 1
    return new_data, num_queries


def load_queries(fp):
    num_queries = 0
    if not fp.exists():
        return defaultdict(list), num_queries

    with open(fp, 'r') as f:
        queries = json.load(f)

    # replaces tuples
    for doc_key, spans in queries.items():
        tuple_spans = []
        for span in spans:
            tuple_spans.append(tuple(span))
        queries[doc_key] = tuple_spans
        num_queries += len(tuple_spans)

    # convert to defaultdict
    queries = defaultdict(list, queries)
    # return queries and # of queried spans
    return queries, num_queries


def output_results(fp, config, cycle, scores):
    results = {
        'seed': config['seed'],
        'strategy': config['strategy'],
        'cycle': cycle,
        'num_spans': config['num_spans'],
        'max_docs': config['max_docs'],
    }
    results.update(scores)
    with open(fp, 'w') as f:
        json.dump(results, f)
    return

def create_cycle_dir(config, idx):
    # store results in {sim_dir} / {seed} / {strategy} / {num_spans}spans-{max_docs}docs / {cycle #}
    spans_docs = f'{config["num_spans"]}spans-{config["max_docs"]}docs'
    cycle_dir = config['sim_dir'] / str(config['seed']) / spans_docs / config['strategy'] / f'cycle{idx}'
    cycle_dir.mkdir(parents=True, exist_ok=True)
    change_log(cycle_dir / 'out.log')
    return cycle_dir


def eval_scores(model, config, data_prefix, preds_file=None):
    """
    Evaluate [model] on test or dev set based on [data_prefix].
    Output F1 and mention detection accuracy.
    """
    logging.info(f'Evaluating on {data_prefix}')
    data = util.load_data(config[f"{data_prefix}_path"])
    evaluator = Predictor(model, data, config["singleton_eval"])
    with torch.no_grad():
        f1 = evaluator.evaluate()
        mentions = evaluator.evaluators[ALL].evaluators[3].get_f1()
    scores = {'f1': f1, 'mentions': mentions}
    if preds_file:
        evaluator.write_preds(preds_file)
        logging.info(f"Wrote preds to {preds_file}")
    return scores

def simulation(config, data_train):
    """
    Run active learning  simulation by sampling data from unlabeled pool,
    using gold data to label sampled spans, and then
    finetuning model on labeled data.
    - [config]: contains all simulation hyperparameters
    - [data_train]: data pool to sample from
    """
    # initialize acquisition model as source model
    logging.info(f'Loading acq. model from {config["src_path"]}')
    acq_model = load_model(config, config['src_path'])
    src_path = config['src_path']

    # no labeled instances
    queries = defaultdict(list)

    for i in range(config['start'], config['num_cycles'] + 1):
        logging.info(f'Cycle {i}')
        total_num_spans = i * config['num_spans']
        cycle_dir = create_cycle_dir(config, i)
        model_file = cycle_dir / 'checkpoint.bin'
        queries_file = cycle_dir / 'queries.json'
        res_dev_file = cycle_dir / 'results_dev.json'
        res_test_file = cycle_dir / 'results_test.json'
        queries_found = False

        try:
            # check if queries for current cycle exist
            new_queries, num_queries = load_queries(queries_file)
            if num_queries < total_num_spans:
                raise ValueError(f'File does not contain enough queries')
            logging.info('Found queries already sampled for this cycle')
            logging.info(f'Loading queries from {queries_file}')
            queries = new_queries
            queries_found = True

        except ValueError:
            # else, sample new queries and concat with old queries
            logging.info(f'Sampling {config["num_spans"]} new queries by {config["strategy"]} for this cycle')
            strategy = config['strategy']

            candidates = \
                sample_spans(acq_model, data_train, queries, config, strategy, cycle=i)
            with open(queries_file, 'w') as f:
                json.dump(queries, f)


        # If model already exists and queries found, don't finetune new model
        if model_file.exists() and queries_found:
            logging.info('Finetuned model for cycle already exists')
            logging.info(f'Loading acq. model from {cycle_dir}')
            util.load_params(acq_model, model_file, "model")
            # evaluate model
            logging.info('Evaluating model')
            preds_file_dev = cycle_dir / 'preds_dev.json'
            scores_dev = eval_scores(acq_model, config, "dev", preds_file_dev)
            logging.info(f'Model has {scores_dev["f1"]:.3f} dev F1')
            output_results(res_dev_file, config, i, scores_dev)
            if config["test_set"]:
                preds_file_test = cycle_dir / 'preds_test.json'
                scores_test = eval_scores(acq_model, config, "test",
                        preds_file_test)
                output_results(res_test_file, config, i, scores_test)
            continue #move on to next cycle


        # get subset of train data for queried spans
        data_query, num_queries  = extract_data(data_train, queries)
        # finetune model on queries
        logging.info(
            f'Finetuning src model on {num_queries} queries from {len(data_query)} docs'
        )
        scores_dev, model = finetune_on_queries(config, data_query, cycle_dir, src_path)
        output_results(res_dev_file, config, i, scores_dev)
        if config["test_set"]:
            preds_file = cycle_dir / 'preds_test.json'
            scores_test = eval_scores(model, config, "test", preds_file)
            output_results(res_test_file, config, i, scores_test)

        logging.info(f'Saved model, queries, scores to {cycle_dir}')
        # set acquisition model to model finetuned on queries
        logging.info(f'Loading acq. model from {cycle_dir} for next cycle')
        util.load_params(acq_model, model_file, "model")

        # remove old model files
        if not config['save_ckpt']:
            logging.info(f'Removing {model_file}')
            model_file.unlink()

def userstudy(config, data_train):
    """
    Update the model based on feedback from user study.
    - [config]: hyperparameters for model fine-tuning
    - [data_train]: data pool to sample from
    """
    def preprocess_data(doc, queries):
        """
        Create a new field in [doc] called [antecedent_map] which processes
        the user-labeled [antecedents]. Add all labeled spans to [queries].
        in queries).
        """
        ante_map = {}
        for entry in doc['antecedents']:
            span = tuple(entry[0])
            if entry[1] == -1:
                label = None
            elif entry[1] == 0:
                label = '0'
            else:
                label = [tuple(entry[1])]
            ante_map[span] = label
        doc['antecedent_map'] = ante_map
        del doc['antecedents']

        # update queries to know what has been queried
        queries[doc['doc_key']] = list(ante_map.keys())
        # return # spans labeled
        return len(ante_map)

    # preprocess antecedents and get queries
    data_fp = config['userstudy'] / 'train_data.jsonl'
    data = []
    queries = defaultdict(list)
    num_queries = 0
    with open(data_fp, 'r') as f:
        for line in f:
            doc = json.loads(line)
            # update doc and queries
            n = preprocess_data(doc, queries)
            num_queries += n
            data.append(doc)
    # finetune model on data
    src_path = config['src_path']
    logging.info(
        f'Finetuning src model on {num_queries} queries from {len(data)} docs'
    )
    scores_dev, model = finetune_on_queries(config, data, config['userstudy'], src_path)

    # test model
    results_fp = config['userstudy'] / 'results_test.json'
    scores_test = eval_scores(model, config, "test")
    output_results(results_fp, config, 1, scores_test)


def write_config(c):
    config = copy.copy(c)
    config_path = config["sim_dir"] / "config.json"
    logging.info(f"Saved at {config_path}")
    config["device"] = str(config["device"])
    config["sim_dir"] = str(config["sim_dir"])
    config["src_path"] = str(config["src_path"])
    with open(config_path, 'w+') as f:
        f.write(json.dumps(config, indent=4))

def change_log(logfile):
    log = logging.getLogger()
    for hdlr in log.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr,logging.FileHandler):
            log.removeHandler(hdlr)
    filehandler = logging.FileHandler(logfile,'a')
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    filehandler.setFormatter(formatter)
    log.addHandler(filehandler)


if __name__ == "__main__":
    config = util.initialize_from_env(use_overrides=True)
    config['src_path'] = pathlib.Path('checkpoint.bin')
    config['sim_dir'] = pathlib.Path(config['simulation_dir']) / config['run_name']
    util.set_seed(config)
    train_pool = util.load_data(config[f'train_path'],
            num_examples=config['num_train_examples'])

    if 'user' in config:
        config['userstudy'] =  \
            pathlib.Path(f'/exp/myuan/userstudy/day1/session{config["session"]}') / config["user"]
        userstudy(config, train_pool)
    else:
        simulation(config, train_pool)
        write_config(config)


