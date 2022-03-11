#!/usr/bin/env python3

from collections import defaultdict
import os

from coref import read_jsonl, write_jsonl


def split_assignments(input_path, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    annotator_hits = defaultdict(list)
    for hit_wrapper in read_jsonl(input_path):
        for annotator in hit_wrapper['assignments'][-1]:
            annotator_hits[annotator].append(hit_wrapper['hit'])

    for (annotator, hits) in annotator_hits.items():
        output_path = os.path.join(output_dir, annotator + '.jsonl')
        write_jsonl(hits, output_path)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Split last round of HIT assignments, producing a separate jsonl file of '
                    'HITs for each annotator.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to input jsonl HIT assignments.')
    parser.add_argument('output_dir',
                        help='Directory where output HITs will be written (using annotator names '
                             'as file names, adding a .jsonl extension).')
    args = parser.parse_args()

    split_assignments(args.input_path, args.output_dir)


if __name__ == '__main__':
    main()
