import pathlib
import json
import copy
import transformers
import pandas as pd
import sys

# have lots of util functions to avoid needing gpu to run script

def flatten(l):
    return [item for sublist in l for item in sublist]

def is_pronoun(span):
    PRONOUNS = ["i", "me", "my", "myself", "you", "your", "yourself", "she", "her", "herself", "he", "him", "his",
            "himself", "it", "its", "we", "our", "they", "them", "their"]
    return span.lower() in PRONOUNS

def load_data(path, num_examples=None):
    if path is None or not path:
        return []
    def load_line(line):
        example = json.loads(line)
        # Need to make antecedent dict
        clusters = [sorted(cluster) for cluster in example["clusters"]]
        antecedent_map = {}
        for cluster in clusters:
            antecedent_map[tuple(cluster[0])] = "0"
            for span_idx in range(1, len(cluster)):
                antecedent_map[tuple(cluster[span_idx])] = [tuple(span) for span in cluster[:span_idx]]
        example["antecedent_map"] = antecedent_map
        return example
    with open(path) as f:
        data = [load_line(l) for l in f.readlines()]
        if num_examples is not None:
            data = data[:num_examples]
        print("Loaded {} examples.".format(len(data)))
        return data

def load_pool(config):
    data = load_data(config['train_path'], config.get('num_train_examples'))
    data_dict = {}
    for doc in data:
        doc_key = doc.pop('doc_key')
        data_dict[doc_key] = doc
    return data_dict

def load_queries(cycle_dir):
    if not (cycle_dir / 'queries.json').exists():
        queries = {}
    else:
        with open(cycle_dir / 'queries.json', 'r') as f:
            queries = json.load(f)
        # replaces with tuples
        for doc_key, spans in queries.items():
            tuple_spans = []
            for span in spans:
                tuple_spans.append(tuple(span))
            queries[doc_key] = tuple_spans
    return queries

def prev_cycle_queries(cycle_dir, num_spans):
    # get queries from previous cycle dir
    cycle_idx = int(cycle_dir.name[5:])
    prev_name = cycle_dir.name[:5] + str(cycle_idx - 1)
    prev_dir = cycle_dir.parents[0] / prev_name
    # prev_queries is {} if prev_dir or its queries doesn't exist
    prev_queries = load_queries(prev_dir)
    return prev_queries


def queries_difference(prev, new):
    diff = {}
    for doc_key in new:
        new_spans = new[doc_key]
        prev_spans = prev.get(doc_key, [])
        diff_spans = list(set(new_spans) - set(prev_spans))
        if len(diff_spans) > 0:
            diff[doc_key] = diff_spans
    return diff

def mark_text(spans, tokens):
    new_tokens = []
    # process each token
    for i in range(len(tokens)):
        # find span openings at position i
        left = [span for span in spans if span[0] == i]
        # sort descending order of second value
        left.sort(key=lambda x: x[1], reverse=True)
        for l in left:
            idx = spans.index(l)
            new_tokens.append(f'<SPAN {idx}>')
        # insert token
        new_tokens.append(tokens[i])
        # find span closings at position i
        right = [span for span in spans if span[1] == i]
        # sort descending order of first value
        right.sort(key=lambda x: x[0], reverse=True)
        for r in right:
            idx = spans.index(r)
            new_tokens.append(f'</SPAN {idx}>')
    return new_tokens


def queries_to_text(queries, data, tokenizer):
    # convert queries to raw text with sampled spans marked
    texts = []
    for doc_key in queries:
        if doc_key not in data:
            raise ValueError('Query doc not in data')
        segments = data[doc_key]['sentences']
        tokens = flatten(segments)
        spans = queries[doc_key]

        new_tokens = mark_text(spans, tokens)

        text = tokenizer.convert_tokens_to_string(new_tokens)
        entry = (doc_key, text, len(spans))
        if len(spans) > 0:
            texts.append(entry)
    return texts

def queries_stats(queries, data):
    # print stats of sampled queries
    tokenizer = transformers.AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    num_pronouns = 0
    num_entities = 0
    num_non = 0
    num_singletons = 0
    for doc_key in queries:
        segments = data[doc_key]['sentences']
        tokens = flatten(segments)
        ante_map = data[doc_key]['antecedent_map']
        spans = queries[doc_key]
        for span in spans:
            text = tokenizer.convert_tokens_to_string(tokens[span[0]:span[1]+1])
            if is_pronoun(text):
                num_pronouns += 1

            if span in ante_map:
                num_entities += 1
                if len(ante_map[span]) == 1:
                    num_singletons += 1
            else:
                num_non += 1
    stats = {'entities': num_entities, 'non-entities': num_non,
            'pronouns':num_pronouns, 'singletons': num_singletons}
    return stats


def output_sim_results(sim_dir, data, config):
    """
    For each simulation cycle subdirectory in [sim_dir], output:
    - [new_queries.txt]: a text file showing spans and docs that were sampled
    - stats of samples queried (if [count] is True)

    Accumulate results (F1 and detection accuracy) over all subdirectories and
    output [results_test.json] in [sim_dir]
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    results = []
    for cycle_dir in sim_dir.glob("**/"):
        print(cycle_dir)

        queries = load_queries(cycle_dir)

        # collect scores
        if (cycle_dir / 'results_test.json').exists():
            with open(cycle_dir / 'results_test.json') as f:
                scores  = json.load(f)

            if scores['num_spans'] == 50 and scores['max_docs'] == 1:
                # count stats for queries
                stats = queries_stats(queries, data)
                scores.update(stats)

            results.append(scores)

        # output new_queries
        print(f'Analyzing {cycle_dir}')
        if len(queries) == 0:
            print('No queries found, skipping')
            continue
        prev_queries = prev_cycle_queries(cycle_dir, config['num_spans'])
        diff_queries = queries_difference(prev_queries, queries)
        texts = queries_to_text(diff_queries, data, tokenizer)
        # sort texts by amount of sampled spans
        texts.sort(key=lambda x:x[2], reverse=True)

        print('Output queries')
        with open(cycle_dir / 'new_queries.txt', 'w') as f:
            total_spans = sum([x[2] for x in texts])
            print(f'{total_spans} spans\t{len(texts)} docs', file=f)
            for text in texts:
                print(f'{text[0]}\t{text[1]}', file=f)


    if len(results) > 0:
        # output scores
        print('Output results')
        df = pd.DataFrame(results).sort_values(by=["seed","num_spans", "max_docs", "strategy", "cycle"])
        df.to_csv(sim_dir / 'results_test.csv', index=False)
    else:
        print('No results')


if __name__ == "__main__":
    # sim_dir should have config.json
    sim_dir = pathlib.Path(sys.argv[1])

    config_file = sim_dir / 'config.json'
    if not config_file.exists():
        raise ValueError('Invalid simulation directory')
    with open(config_file, 'r') as f:
        config = json.load(f)

    # load data pool as dict of doc_keys
    data = load_pool(config)
    output_sim_results(sim_dir, data, config)








