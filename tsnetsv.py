#!/usr/bin/env python

'''
Use t-SNE to generate TSV output files for downstream processing.

Note: Largely follows Turian's test.py

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2012-09-11
'''

from argparse import ArgumentParser, FileType
from itertools import izip
from math import sqrt
from sys import stderr, stdin, stdout

from numpy import array

from calc_tsne import tsne

def _argparser():
    argparser = ArgumentParser('Generate TSV output for t-SNE')
    argparser.add_argument('-i', '--input', type=FileType('r'), default=stdin,
            help='input source (default: stdin)')
    argparser.add_argument('-o', '--output', type=FileType('w'), default=stdout,
            help='output (default: stdout)')
    argparser.add_argument('-w', '--whitespace-cells', action='store_true',
            help='input cells separated by whitespace (default: tabs)')
    return argparser

def main(args):
    argp = _argparser().parse_args(args[1:])

    ### Build the numpy array to be used by the library
    if not argp.whitespace_cells:
        cell_sep = '\t'
    else:
        cell_sep = None
    # Keep the tokens along to align them later
    toks = []
    toks_vals = []
    for line_num, line in enumerate((l.rstrip('\n') for l in argp.input),
            start=1):
        try:
            tok, vals_str = line.split(cell_sep, 1)
            tok_vals = [float(v) for v in vals_str.split(cell_sep)]
        except ValueError:
            print >> stderr, ('ERROR: Failed to read input line %s "%s"'
                    ) % (line_num, line, )
            print >> stderr, ('ERROR: Is the input perhaps separated by '
                    'spaces instead of tabs? If so, try the -w flag')
            return -1

        toks.append(tok)
        toks_vals.append(tok_vals)
    toks_vals_array = array(toks_vals)
    del toks_vals

    ### Perform t-SNE (this is heavy stuff)
    # TODO: A majority of the below arguments could be program arguments
    tsne_output = tsne(toks_vals_array, no_dims=2, perplexity=30, initial_dims=30)
    del toks_vals_array

    ### Re-format and produce TSV output
    # Normalise the dimensions onto the range [0, 1]
    max_x = max(x for x, _ in tsne_output)
    min_x = min(x for x, _ in tsne_output)
    max_y = max(y for _, y in tsne_output)
    min_y = min(y for _, y in tsne_output)

    x_interval = sqrt((max_x - min_x) ** 2)
    y_interval = sqrt((max_y - min_y) ** 2)

    normalised_tsne_output = (((x - min_x) / x_interval,
        (y - min_y) / y_interval) for x, y in tsne_output)

    ### And write it all out
    for token, point in izip(toks, normalised_tsne_output):
        argp.output.write('{}\t{}\t{}\n'.format(token, point[0], point[1]))

    return 0

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
