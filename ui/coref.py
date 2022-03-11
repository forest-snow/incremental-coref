#!/usr/bin/env python3

import json
import re
import logging
from random import shuffle, random, choices
from collections import defaultdict, Counter


BEGIN_DOCUMENT_RE = re.compile(r'^#begin document (?P<doc>[^ ]+)(?:; part (?P<part>\d+))?$')


def make_span(start_position, end_position):
    return (start_position - 1, end_position)


class SyntaxTreeNode(object):
    def __init__(self, tag=None, position=None, text=None, parent=None):
        self.tag = tag
        self.position = position
        self.text = text
        self.parent = parent
        self.children = []

    def get_span(self):
        node_positions = [n.position for n in self.get_leaves()]
        return make_span(min(node_positions), max(node_positions))

    def get_sorted_leaves(self):
        return sorted(self.get_leaves(), key=lambda n: n.position)

    def get_nodes(self):
        return [self] + self.get_descendants()

    def get_leaves(self):
        return [n for n in self.get_nodes() if n.position is not None]

    def get_descendants(self):
        return [n for c in self.children for n in c.get_nodes()]

    def get_tokens(self):
        return [n.text for n in self.get_sorted_leaves()]

    def __str__(self):
        position = (self.position if self.position is not None else '#')
        tag = (self.tag if self.tag is not None else '^')
        text = (self.text if self.text is not None else '')
        return '{} {} {}'.format(position, tag, text) + (
            '\n- ' + '\n- '.join(str(c).replace('\n', '\n  ') for c in self.children)
            if self.children
            else '')

    @classmethod
    def from_uds_graph(cls, uds_graph):
        nodes = uds_graph.syntax_nodes
        subtrees = dict(
            (
                node_name,
                SyntaxTreeNode(
                    tag=node_data['xpos'],
                    position=node_data['position'],
                    text=node_data['form'])
            )
            for (node_name, node_data)
            in nodes.items())

        # sorting edges by to-node before iteration ensures each node's
        # children are sorted by to-node
        edges = sorted(uds_graph.syntax_edges(), key=lambda e: nodes[e[1]]['position'])
        root_name = None
        for (from_node_name, to_node_name) in edges:
            if from_node_name.endswith('-root-0'):
                root_name = to_node_name
            else:
                subtrees[to_node_name].parent = subtrees[from_node_name]
                subtrees[from_node_name].children.append(subtrees[to_node_name])

        return subtrees[root_name]

    @classmethod
    def from_ptb_str(cls, ptb_str):
        ptb_str = ptb_str.strip()

        stack = []
        root = None

        while ptb_str:
            if ptb_str.startswith('('):
                ptb_str = ptb_str[1:].lstrip()
                node = SyntaxTreeNode()
                if not ptb_str.startswith('('):
                    (tag, ptb_str) = ptb_str.split(maxsplit=1)
                    node.tag = tag

                if root is None:
                    root = node
                if stack:
                    node.parent = stack[-1]
                    stack[-1].children.append(node)

                stack.append(node)

            elif ptb_str.startswith(')'):
                ptb_str = ptb_str[1:].lstrip()
                stack.pop()

            else:
                text_end = ptb_str.index(')')
                text = ptb_str[:text_end].rstrip()
                ptb_str = ptb_str[text_end:]

                stack[-1].text = text

        if stack:
            raise Exception('PTB data consumed but stack not empty')

        stack.append(root)
        position = 1

        while stack:
            node = stack.pop()

            if node.tag == '-NONE-':
                if node.children:
                    raise Exception('-NONE- PTB node has children')
                # leaf node is empty, remove empty ancestors recursively
                while not node.children:
                    node.parent.children.remove(node)
                    node = node.parent

            elif node.text is not None:
                if node.children:
                    raise Exception(' PTB node with text has children')
                node.text = node.text.replace('-LRB-', '(').replace('-RRB-', ')')
                node.position = position
                position += 1

            else:
                for child in node.children[::-1]:
                    stack.append(child)

        return root


def get_semantics_span(uds_graph, semantics_node_name):
    try:
        tokens_by_position = uds_graph.span(semantics_node_name)
    except ValueError:
        return None
    else:
        return make_span(min(tokens_by_position), max(tokens_by_position))


def get_span_words(span, words):
    return [words[i] for i in range(*span)]


def human_format_span(span, words):
    return '({:>{width}}, {:>{width}}): {}'.format(
        span[0],
        span[1],
        ' '.join(get_span_words(span, words)), width=len(str(len(words))))


def human_format_cluster(cluster, words):
    return ' | '.join(human_format_span(span, words) for span in cluster)


def make_word_num_cluster_triples_map(clusters):
    '''
    Return map from word numbers to lists representing the clusters they
    appear in.  Each list contains one triple for each containing
    cluster: (cluster number, is beginning of span?, is end of span?)
    '''
    m = defaultdict(list)
    for (cluster_num, span) in enumerate(clusters):
        for word_num in range(span[0], span[1] + 1):
            m[word_num].append((cluster_num, word_num == span[0], word_num == span[1]))
    return m


def format_cluster_triple(cluster_triple):
    return '{}{}{}'.format('(' if cluster_triple[1] else '',
                           cluster_triple[0],
                           ')' if cluster_triple[2] else '')


def parse_cluster_triple(cluster_str):
    return (int(cluster_str.strip('()')),
            cluster_str.startswith('('),
            cluster_str.endswith(')'))


def write_conll_predictions(docs, path):
    with open(path, mode='w', encoding='utf-8') as f:
        for doc in docs:
            doc_id = doc['doc_key']
            f.write('#begin document ({}); part 0\n'.format(doc_id))

            word_num_cluster_triples = make_word_num_cluster_triples_map(doc['clusters'])
            word_num = 0

            for sentence in doc['sentences']:
                for _ in sentence:
                    f.write('{} {}\n'.format(doc_id, '|'.join(
                        format_cluster_triple(t) for t in word_num_cluster_triples[word_num])))

                    word_num += 1
                f.write('\n')

            f.write('#end document\n')


def write_jsonl(objects, path):
    with open(path, 'w') as f:
        for obj in objects:
            f.write(json.dumps(obj) + '\n')


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def iter_conll_doc_lines(path):
    with open(path, encoding='utf-8') as f:
        doc_id = None
        part_id = None
        doc_lines = []

        for line in f:
            m = BEGIN_DOCUMENT_RE.match(line)
            if m is not None:
                doc_id = m.group('doc')
                part_id = m.group('part')
                doc_lines = []

            elif line.startswith('#end document'):
                yield (doc_id, part_id, doc_lines)
                doc_id = None
                part_id = None
                doc_lines = []

            else:
                doc_lines.append(line)

        if doc_id is not None:
            raise Exception('reading stopped before doc {} part {} ended'.format(doc_id, part_id))


def iter_conll(path):
    for (doc_id, part_id, doc_lines) in iter_conll_doc_lines(path):
        word_num = 0
        sentences = []
        sentence = []
        clusters = defaultdict(list)
        cluster_span_starts = defaultdict(list)
        for line in doc_lines:
            if line.strip():
                pieces = line.strip().split()
                sentence.append(pieces[3])
                if pieces[-1] != '-':
                    for s in pieces[-1].split('|'):
                        (cluster_num, span_start, span_end) = parse_cluster_triple(s)
                        if span_start:
                            cluster_span_starts[cluster_num].append(word_num)
                        if span_end:
                            clusters[cluster_num].append(
                                [cluster_span_starts[cluster_num].pop(), word_num])

                word_num += 1

            else:
                sentences.append(sentence)
                sentence = []

        if sentence:
            raise Exception('doc {} ended before sentence ended'.format(doc_id))
        if any(cluster_span_starts.values()):
            raise Exception('doc {} ended before all cluster spans ended'.format(doc_id))

        yield dict(
            doc_key=('{}_{}'.format(doc_id, part_id) if part_id else doc_id),
            sentences=sentences,
            clusters=list(clusters.values()))


def iter_sentences_as_docs(sentences):
    document = None
    for sentence in sentences:
        if document is None or sentence['document_id'] != document['document_id']:
            if document is not None:
                yield document
            document = dict(document_id=sentence['document_id'], sentences=[])
        del sentence['document_id']
        document['sentences'].append(sentence)

    if document is not None:
        yield document


def compute_last_round_assignment_divergence(doc_wrappers):
    annotator_num_sentences = Counter()
    for doc_wrapper in doc_wrappers:
        for annotator in doc_wrapper['assignments'][-1]:
            annotator_num_sentences[annotator] += len(doc_wrapper['doc'])

    return max(annotator_num_sentences.values()) - min(annotator_num_sentences.values())


def compute_quartiles(x):
    s = sorted(x)
    return [
        s[0],
        s[round(0.25 * (len(s) - 1))],
        s[round(0.50 * (len(s) - 1))],
        s[round(0.75 * (len(s) - 1))],
        s[-1]
    ]


class AssignmentSampler(object):
    def __init__(self):
        self.num_trials = 0
        self.trial_divergences = []

    def loop(self, *args, **kwargs):
        while not self.run_trial(*args, **kwargs):
            pass

    def run_trial(self, doc_wrappers, annotators, redundancy, tolerance):
        annotators = set(annotators)

        for doc_wrapper in doc_wrappers:
            prev_annotators = set(
                annotator
                for round_assignments in doc_wrapper['assignments'][:-1]
                for annotator in round_assignments)
            doc_wrapper['assignments'][-1] = list(annotators.difference(prev_annotators))

        annotator_counts = Counter(
            annotator
            for doc_wrapper in doc_wrappers
            for annotator in doc_wrapper['assignments'][-1])
        annotator_weights = dict(
            (annotator, 1 / count)
            for (annotator, count)
            in annotator_counts.items())

        for doc_wrapper in doc_wrappers:
            candidate_annotators = doc_wrapper['assignments'][-1]
            doc_wrapper['assignments'][-1] = []
            for _ in range(redundancy):
                [annotator] = choices(
                    candidate_annotators,
                    weights=[annotator_weights[a] for a in candidate_annotators])
                doc_wrapper['assignments'][-1].append(annotator)
                candidate_annotators.remove(annotator)

        div = compute_last_round_assignment_divergence(doc_wrappers)
        success = (div <= tolerance)

        self.num_trials += 1
        self.trial_divergences.append(div)

        if not success and self.num_trials % 1000 == 0:
            logging.info('Failed {} trials (divergence quartiles {}; tolerance {})'.format(
                self.num_trials,
                '[{} {} {} {} {}]'.format(*compute_quartiles(self.trial_divergences)),
                tolerance))

        return success


def make_assignments(doc_wrappers, annotators, redundancy, tolerance):
    # doc_wrappers is an iterable of dictionaries with the keys 'doc'
    # and 'assignments', where 'doc' is a list of HITs to be assigned
    # together and 'assignments' is a list of lists of annotator
    # names, where each nested list represents a round of annotation.

    doc_wrappers = list(doc_wrappers)

    # add empty new round

    for doc_wrapper in doc_wrappers:
        doc_wrapper['assignments'].append([])

    # assign annotators to HITs (docs) in new round

    AssignmentSampler().loop(
        doc_wrappers,
        annotators=annotators,
        redundancy=redundancy,
        tolerance=tolerance)

    # shuffle doc wrappers to mitigate ordering effects

    shuffle(doc_wrappers)

    # generate HIT wrappers with new assignments

    for doc_wrapper in doc_wrappers:
        for hit in doc_wrapper['doc']:
            yield dict(
                hit=hit,
                assignments=doc_wrapper['assignments'])
