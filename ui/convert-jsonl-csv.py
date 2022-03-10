#!/usr/bin/env python3

import argparse
import csv
import json
from glob import glob


DEFAULT_COLUMN_NAME = 'json_data'


def csv_to_jsonl(csv_path, jsonl_path, column_name, keep=False, extra_parse_columns=None):
    if extra_parse_columns is None:
        extra_parse_columns = []

    with open(jsonl_path, "w") as jsonl_fh, \
            open(csv_path, "r", encoding="utf-8") as csv_fh:

        reader = csv.DictReader(csv_fh)
        for row in reader:
            if keep:
                for parse_column in [column_name] + list(extra_parse_columns):
                    row[parse_column] = json.loads(row[parse_column])
                json_str = json.dumps(dict(row))
            else:
                json_str = row[column_name]

            jsonl_fh.write(json_str + '\n')


def jsonl_to_csv(jsonl_path, csv_path, column_name):
    with open(jsonl_path, "r") as jsonl_fh, \
            open(csv_path, "w", encoding="utf-8") as csv_fh:

        fieldnames = [column_name]
        writer = csv.DictWriter(csv_fh, fieldnames, lineterminator='\n')
        writer.writeheader()
        for line in jsonl_fh:
            writer.writerow({column_name: line.strip()})


def main():
    import sys
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''\
Convert between CSV and jsonl (one-JSON-object-per-line) formats.

Provided an input file path and an output file path,
convert the input file contents to the other format and save the
converted contents to the output file path.  The input and output
formats will be guessed from the file extensions (`.csv` or `.jsonl`)
and can also be specified explicitly with a command-line flag
(`-d c2j` or `-d j2c`).  For example,
`{script} data.jsonl data.csv` or `{script} -d j2c data.in data.out`
for jsonl-to-csv or
`{script} data.csv data.jsonl` or `{script} -d c2j data.in data.out`
for csv-to-jsonl.
'''.format(script=sys.argv[0]),
    )
    parser.add_argument('input_path')
    parser.add_argument('output_path', nargs='?', default=None,
                        help='If not set, computed from input_path by replacing a '
                             '.csv or .jsonl extension with .jsonl or .csv (respectivley)')
    parser.add_argument('-d', '--direction', choices=('j2c', 'c2j'),
                        help="Direction of conversion (default: infer from file extensions)")
    parser.add_argument('-c', '--column-name', default=DEFAULT_COLUMN_NAME,
                        help="Name of input/output CSV column containing JSON data")
    parser.add_argument('-k', '--keep', action='store_true',
                        help="When converting csv to jsonl, retain the rest of the row data "
                             "as a dictionary wrapping the extracted json.")
    parser.add_argument('-p', '--extra-parse-column', action='append', dest='extra_parse_columns',
                        help="When converting csv to jsonl and `keep` is specified, "
                             "parse the specified column as "
                             "JSON in addition to the one specified by `column-name`.  "
                             "This argument may be used multiple times.")
    parser.add_argument('-g', '--glob', action='store_true',
                        help='Interpret input_path as a glob and convert all matching files '
                             '(output_path must not be set)')
    args = parser.parse_args()

    if args.glob and args.output_path is not None:
        raise Exception('glob is set but output_path is also specified')
    input_paths = (glob(args.input_path) if args.glob else [args.input_path])

    for input_path in input_paths:
        convert_jsonl_csv(input_path=input_path,
                          output_path=args.output_path,
                          direction=args.direction,
                          column_name=args.column_name,
                          keep=args.keep,
                          extra_parse_columns=args.extra_parse_columns)


def convert_jsonl_csv(input_path, output_path=None, direction=None,
                      column_name=DEFAULT_COLUMN_NAME, keep=False, extra_parse_columns=None):
    input_path_l = input_path.lower()

    if output_path is None:
        if input_path_l.endswith('.csv'):
            output_path = input_path[:-len('.csv')] + '.jsonl'
        elif input_path_l.endswith('.jsonl'):
            output_path = input_path[:-len('.jsonl')] + '.csv'
        else:
            raise Exception('output_path not set and input_path extension not recognized')

    output_path_l = output_path.lower()

    if direction is None:
        if input_path_l.endswith('.csv') and not output_path_l.endswith('.csv'):
            direction = 'c2j'
        elif not input_path_l.endswith('.jsonl') and output_path_l.endswith('.jsonl'):
            direction = 'c2j'
        elif input_path_l.endswith('.jsonl') and not output_path_l.endswith('.jsonl'):
            direction = 'j2c'
        elif not input_path_l.endswith('.csv') and output_path_l.endswith('.csv'):
            direction = 'j2c'

    if direction == 'c2j':
        csv_to_jsonl(input_path, output_path, column_name=column_name,
                     keep=keep, extra_parse_columns=extra_parse_columns)
    elif direction == 'j2c':
        jsonl_to_csv(input_path, output_path, column_name=column_name)
    else:
        raise Exception('conversion direction was not specified and could not be inferred')


if __name__ == "__main__":
    main()
