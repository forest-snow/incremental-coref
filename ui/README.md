# Coref annotation

This repository contains an Amazon Mechanical Turk/Turkle template (`active-learning-hit.html`) and tools for annotating incremental coreference links, that is, links from an entity mention back to a previous mention of the same entity in the document.

## Assigning HITs

When using a fixed pool of annotators, to assign each annotator a
similar amount of work, use `make-assignments.py` and
`split-assignments.py`.  `make-assignments.py` assigns each line of the
original data to an annotator and `split-assignments.py` takes the
output and produces a separate file for each annotator's HITs.

## Serving the interface locally

The provided HTML template is compatible with Amazon Mechanical Turk.
It can also be served locally, for development, using [prototurk](https://github.com/hltcoe/prototurk) (or, if you want to do crowd annotation locally, [Turkle](https://github.com/hltcoe/turkle)):

Install prototurk with

```bash
pip install prototurk
```

And then start a local development server with (for example):

```bash
prototurk active-learning-hit.html data/pilot-hits.csv
```

## Annotation

For annotation, we generally use a JSON lines file format (one JSON object per line).  Mechanical Turk uses CSV input and output, so a Python script `convert-jsonl-csv.py` is provided for converting between the two formats.  Use `convert-jsonl-csv.py -h` to see the usage.

### Input

For the basic coref annotation protocol (`base-hit.html`), each input line has the following format (here a single line has been expanded onto multiple lines and indented to illustrate the structure):

```json
{
  "sentences": [
    ["john", "crashed", "his", "car", "."],
    ["the", "crash", "occurred", "in", "catonsville", "."],
    [],
    ["the", "end", "."]
  ],
  "querySpans": [
    { "sentenceIndex": 1, "startToken": 0, "endToken": 2 },
    { "sentenceIndex": 1, "startToken": 4, "endToken": 5 }
  ],
  "candidateSpans": [
    { "sentenceIndex": 0, "startToken": 0, "endToken": 1 },
    { "sentenceIndex": 0, "startToken": 1, "endToken": 2 },
    { "sentenceIndex": 0, "startToken": 1, "endToken": 4 },
    { "sentenceIndex": 0, "startToken": 2, "endToken": 4 },
    { "sentenceIndex": 0, "startToken": 3, "endToken": 4 },
    { "sentenceIndex": 1, "startToken": 0, "endToken": 2 },
    { "sentenceIndex": 1, "startToken": 1, "endToken": 2 },
    { "sentenceIndex": 1, "startToken": 4, "endToken": 5 }
  ]
}
```

The contents of `sentences` consist of a list of sentences, each of which is represented by a list of words (strings).

In general, text spans are represented as a three-element dictionaries with the following entries:

* `sentenceIndex`: the zero-based index of the sentence in `sentences`
* `startToken`, `endToken`: a zero-based exclusive range corresponding to the words of the span within the specified sentence 

Each entry of `querySpans` represents a span in the sentences to be annotated.

The content of `candidateSpans` represents the spans in the sentences that can be submitted as answers.

### Output

Each line of the output JSON lines data, extracted and converted from the CSV column `Answer.answer_spans` in the Mechanical Turk output, consists of a list of answer spans and a list of timestamped interface events for analysis:

```json
{
  "answerSpans": [
    {
      "querySpan": { "sentenceIndex": 1, "startToken": 0, "endToken": 2 },
      "sentenceIndex": 0,
      "startToken": 1,
      "endToken": 4,
      "notPresent": false,
      "status": "ok"
    },
    {
      "querySpan": { "sentenceIndex": 1, "startToken": 4, "endToken": 5 },
      "sentenceIndex": -1,
      "startToken": -1,
      "endToken": -1,
      "notPresent": true,
      "status": "ok"
    }
  ],
  "events": [
    {"timestampSeconds": 1619561322, "type": "start"},
    {"timestampSeconds": 1619561322, "type": "selectQuery", "queryIndex": 0},
    {"timestampSeconds": 1619561347, "type": "lastSelectAnswer", "queryIndex": 0},
    {"timestampSeconds": 1619561349, "type": "selectQuery", "queryIndex": 1},
    {"timestampSeconds": 1619561352, "type": "lastSelectAnswer", "queryIndex": 1},
    {"timestampSeconds": 1619561354, "type": "selectQuery", "queryIndex": 2},
    {"timestampSeconds": 1619561418, "type": "lastSelectAnswer", "queryIndex": 2},
    {"timestampSeconds": 1619561420, "type": "submit"}
  ],
  "elapsedSeconds": 98
}
```

The value of `querySpan` represents a given answer span's corresponding query span and is copied from the input data; the outer sentence index, start token, and end token represent the answer span.

If there is no answer for a given query span, the answer span indices are all set to `-1` and `notPresent` is set to `true`.
