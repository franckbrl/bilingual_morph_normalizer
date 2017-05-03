#!/usr/bin/python3
# -*- coding: utf-8 -*-

##################################################
# Franck Burlot <franck.burlot@limsi.fr>
# 2017 - LIMSI-CNRS
##################################################

"""
Post-process the SMT output. OOVs are translated
as lemmas and a sequence of tags (using the separator
given as argument). Extract and return the lemma.

USAGE: cat output | python3 normal_postproc.py [separator] > output.pproc
"""

import sys

# Check for separator in arguments
if len(sys.argv) == 2:
    sep = sys.argv[1]
else:
    sep = '+'

for line in sys.stdin:
    line_out = []
    line = line.rstrip().split()
    for word in line:
        if sep in word:
            word = word.split(sep)[0]
        line_out.append(word)
    print(' '.join(line_out))
