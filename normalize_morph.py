#!/usr/bin/python3
# -*- coding: utf-8 -*-

##################################################
# Franck Burlot <franck.burlot@limsi.fr>
# LIMSI-CNRS - 2017
##################################################

import argparse
import pickle

from collections import defaultdict


def clean_word(seq):
    """
    If the separator is in the word,
    replace it by 'rep'.
    """
    rep = '-'
    if type(seq) == list:
        lem = seq[0]
        lem = lem.replace(args.sep, rep)
        return [lem] + seq[1:]
    else:
        return seq.replace(args.sep, rep)


parser = argparse.ArgumentParser(description = \
"""
Normalize monolingual data according to the normalization
model (learnt with train_bmn.py).
""")
parser.add_argument('-i', '--input', dest='input', nargs='?', type=argparse.FileType('r'),
                    help="input file with monolingual data to normalize")
parser.add_argument('-n', '-normal-model', dest='norm', type=str, default='norm_model.pkl',
                    help="pickle file where normalization model were dumped")
parser.add_argument('-o', '--ouput', dest='output', nargs='?', type=argparse.FileType('w'),
                    help="output file with normalized data")
parser.add_argument('-f', '--forms', dest='forms', nargs='?', type=argparse.FileType('r'),
                    help="additional input file with word forms")
parser.add_argument('-s', '--sep', dest='sep', type=str, default='+',
                    help="separator between attributes")
parser.add_argument('-l', '-lemma-limit', dest='lem_limit', type=int, default=100,
                    help="set n most frequent words for lexicalized model.")
parser.add_argument('-use-forms', dest='use_forms', action='store_true',
                    help="use word forms for frequent words (no normalization)")
args = parser.parse_args()

# Get normalization schemata
pos_model, lem_freq = pickle.load( open( args.norm, "rb" ) )

# Build form-to-class mappings
pos_map = {}
for pos, clusters in pos_model.items():
    pos_map[pos] = {}
    for i, cluster in enumerate(clusters):
        for form in cluster:
            pos_map[pos][form] = str(i)

# Delexicalized normalization
if not args.use_forms:
    # Normalize
    for line in args.input:
        sent_out = []
        line = line.rstrip().split('\t')
        for word in line:
            word = word.split()
            lem  = ' '.join(word[:2])
            pos  = word[1]
            form = ' '.join(word[2:])
            if pos in pos_map and form in pos_map[pos]:
                lem = clean_word(lem)
                normalized = args.sep.join( lem.split() + [pos_map[pos][form]] )
                sent_out.append(normalized)
            else:
                word = clean_word(word)
                sent_out.append(args.sep.join(word))
        # Ouput sentence
        args.output.write(' '.join(sent_out) + '\n')

# Output forms for most frequent lemmas
else:
    # Get most frequent lemmas
    lem_freq = lem_freq[:args.lem_limit+1]
    # Normalize
    for line_norm, line_form in zip(args.input, args.forms):
        sent_out = []
        line_norm = line_norm.rstrip().split('\t')
        line_form = line_form.split()
        for word, full_form in zip(line_norm, line_form):
            word = word.split()
            lem  = ' '.join(word[:2])
            pos  = word[1]
            form = ' '.join(word[2:])
            full_form = full_form.rstrip()
            # The lemma is above the frequency limit
            # output word form.
            if lem in lem_freq:
                full_form = clean_word(full_form)
                sent_out.append(full_form)
                continue
            if pos in pos_map and form in pos_map[pos]:
                lem = clean_word(lem)
                normalized = args.sep.join( lem.split() + [pos_map[pos][form]] )
                sent_out.append(normalized)
            else:
                word = clean_word(word)
                sent_out.append(args.sep.join(word))
        # Ouput sentence
        args.output.write(' '.join(sent_out) + '\n')
