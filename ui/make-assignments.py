#!/usr/bin/env python3

import logging

from coref import read_jsonl, write_jsonl, make_assignments


DEFAULT_REDUNDANCY = 1
DEFAULT_TOLERANCE = 10


def load_doc_wrappers(hits, annotators, resume, redundancy):
    # convert-sentences-to-docs.py does similar processing;
    # abstract it?

    doc_wrappers = dict()
    for hit in hits:
        if resume:
            (hit, hit_assignments) = (hit['hit'], hit['assignments'])
        else:
            hit_assignments = []

        num_prev_assignments = sum(len(round_assignments) for round_assignments in hit_assignments)
        if redundancy + num_prev_assignments > len(annotators):
            raise Exception(
                '{}x redundancy (and {} existing assignments) but only {} annotators'.format(
                    redundancy, num_prev_assignments, len(annotators)))

        doc_id = hit['documentId']
        if doc_id in doc_wrappers:
            if hit_assignments != doc_wrappers[doc_id]['assignments']:
                raise Exception('assignments differ between sentences of doc {}'.format(doc_id))
            doc_wrappers[doc_id]['doc'].append(hit)
        else:
            doc_wrappers[doc_id] = dict(doc=[hit], assignments=hit_assignments)

    return doc_wrappers.values()


def load_doc_wrappers_and_make_assignments(hits, annotators, resume=False,
                                           redundancy=DEFAULT_REDUNDANCY,
                                           tolerance=DEFAULT_TOLERANCE):
    # `hits` can be a list of HIT dictionaries (if resume is false)
    # or a list of HIT wrapper dictionaries (if resume is true)

    # load input doc wrappers

    return make_assignments(
        load_doc_wrappers(
            hits,
            annotators=annotators,
            resume=resume,
            redundancy=redundancy),
        annotators=annotators,
        redundancy=redundancy,
        tolerance=tolerance)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Generate random assignments for jsonl HIT data.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to jsonl HIT data.')
    parser.add_argument('output_path', help='Path to output (wrapped) jsonl HIT data.')
    parser.add_argument('--resume', action='store_true',
                        help='Treat input file as existing assignments and add redundancy.')
    parser.add_argument('--annotators', nargs='+', help='Names of annotators.')
    parser.add_argument('--redundancy', type=int, default=DEFAULT_REDUNDANCY,
                        help='Number of times to annotate each item.')
    parser.add_argument('--tolerance', type=int, default=DEFAULT_TOLERANCE,
                        help='Allowable difference in number of items assigned to each annotator.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    write_jsonl(
        load_doc_wrappers_and_make_assignments(
            read_jsonl(args.input_path),
            annotators=args.annotators,
            resume=args.resume,
            redundancy=args.redundancy,
            tolerance=args.tolerance),
        args.output_path)


if __name__ == '__main__':
    main()
