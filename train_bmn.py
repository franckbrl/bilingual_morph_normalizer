#!/usr/bin/python3
# -*- coding: utf-8 -*-

##################################################
# Franck Burlot <franck.burlot@limsi.fr>
# LIMSI-CNRS - 2017
##################################################

import sys
import math
import argparse
import pickle
import operator

from collections import defaultdict
from collections import Counter
from itertools import combinations

import numpy as np


class Distributions():
    """
    Store unigram probability of a source word
    p(f), entropy of the distribution over
    target words H(e|f) and counts c(e|f).
    Keep the list of forms for each PoS, as
    well as the lemma unigram probabilities

    :param type: dict(dict([float, float, Counter]))
    :pos_forms type: dict(tuple)
    :unig_lem type: list(tuple(str, float))
    """
    def __init__(self, src, trg, ali, rev, min_freq, form_min, pos):
        self.min_lem_freq  = min_freq
        self.min_form_freq = form_min
        self.required_pos  = pos.split('-')
        self.collect_counts(src, trg, ali, rev)

    def collect_counts(self, src, trg, ali, rev):
        """
        Compute p(f), H(e|f) and c(e|f) and store them.
        """
        # Parameter storage
        self.param = defaultdict(lambda: defaultdict(lambda: []))
        # Temporary storage
        cnt_bilg   = defaultdict(lambda: defaultdict(lambda: 0))
        cnt_unig   = defaultdict(lambda: 0)
        lemmas     = defaultdict(lambda: 0)
        # Target words are stored as arbitrary numbers (IDs).
        self.trg_ids = {}
        self.trg_ind = 0
        # Form storage for each PoS (before filtering
        # out lemmas under frequency threshold, since
        # we want as many forms as possible)
        self.pos_forms = defaultdict(lambda: set())

        # Get counts from sentences
        for F, E, A in zip(src, trg, ali):
            F = F.rstrip().split('\t')
            E = E.split()
            if rev:
                A = [(int(y),int(x)) for x,y in [pair.split('-') for pair in A.split()]]
            else:
                A = [(int(x),int(y)) for x,y in [pair.split('-') for pair in A.split()]]
            for pair in A:
                try:
                    f = F[pair[0]]
                except IndexError:
                    sys.stderr.write("Unsuited alignments. Maybe reverse them? (-r)\n")
                    raise
                if not self.required(f):
                    continue
                e = self.get_trg_id(E[pair[1]])
                cnt_bilg[f][e] += 1
                cnt_unig[f]    += 1
                # Collect lemmas
                lem = ' '.join(f.split()[:2])
                lemmas[lem] += 1
        del self.trg_ids
        del self.trg_ind

        # Filter out lemmas and forms under frequency minimum
        self.unig_lem = defaultdict(lambda: 0)
        lem_id        = defaultdict(lambda: [])
        for f in cnt_unig:
            lem = ' '.join(f.split()[:2])
            self.unig_lem[lem] += cnt_unig[f]
            lem_id[lem].append(f)
        for lem in self.unig_lem:
            if self.unig_lem[lem] < self.min_lem_freq:
                for f in lem_id[lem]:
                    del cnt_unig[f], cnt_bilg[f]
            # The lemma is kept, remove its forms below threshold
            # (these forms will be dealt with at PoS level).
            else:
                for f in lem_id[lem]:
                    if cnt_unig[f] < self.min_form_freq:
                        del cnt_unig[f], cnt_bilg[f]
                    else:
                        # Collect word forms
                        pos  = f.split()[1]
                        form = ' '.join(f.split()[2:])
                        self.pos_forms[pos].add(form)
        del lem_id

        # Normalize and store lemma probabilities
        self.unig_lem = norm_cnts(self.unig_lem)
        self.unig_lem = sorted(self.unig_lem.items(), key=operator.itemgetter(1), reverse=True)

        # Normalize unigram counts
        prob_unig = norm_cnts(cnt_unig)
        del cnt_unig
        # Get probability distribution p(e|f)
        # and compute local entropy (for one f)
        prob_bilg = {}
        local_entropy = defaultdict(lambda: None)
        for f in cnt_bilg:
            prob_bilg[f] = norm_cnts(cnt_bilg[f])
            local_entropy[f] = compute_loc_entropy(prob_bilg[f].values())

        # Build storage in param: [p(f), H(e|f), c(e|f)]
        for f in prob_unig:
            f = f.split()
            lemma_pos = ' '.join(f[:2])
            tags = ' '.join(f[2:])
            f = ' '.join(f)
            # Add unigram probability
            self.param[lemma_pos][tags].append(prob_unig[f])
            # Add local entropy
            self.param[lemma_pos][tags].append(local_entropy[f])
            # Add distribution c(e|f)
            dist = Counter(cnt_bilg[f])
            self.param[lemma_pos][tags].append(dist)
        del prob_unig, local_entropy, cnt_bilg
        # Storage of (sorted) forms for each tag
        for pos in list(self.pos_forms.keys()):
            self.pos_forms[pos] = tuple(sorted(self.pos_forms[pos]))
        self.pos_forms = dict(self.pos_forms)

    def get_trg_id(self, w):
        """
        Return the ID of the target word
        (create it if needed).
        """
        if w not in self.trg_ids:
            self.trg_ind += 1
            self.trg_ids[w] = self.trg_ind
        return self.trg_ids[w]

    def required(self, f):
        """
        Return True if the PoS of the word
        must be taken into account.
        """
        pos = f.split()[1]
        if pos in self.required_pos or self.required_pos == ['']:
            return True
        else:
            return False

    def get_param(self, lem, i):
        """
        Get p(f), H(e|f) and c(e|f), given a lemma and a form id.
        """
        pos  = lem.split()[1]
        form = self.pos_forms[pos][i]
        return self.param[lem][form]


class IgMatrix():
    """
    Create an initialized matrix for each lemma.

    4 dictionaries contain, for one PoS as key:
    - the parameters of the lemma forms
    - the ids of the forms
    - the IG matrix
    - the merge matrix (containing parameters of newly merged words)

    :type param: dict(dict(list(list)))
    :type ids: dict(list(list(int)))
    :type matrices: dict(dict(numpy.ndarray(float)))
    :type saved_merges: dict(dict(numpy.ndarray(list)))
    """
    def __init__(self, model):
        self.lemmas       = [l for l in model.param]
        self.matrices     = defaultdict(lambda: defaultdict(lambda: []))
        self.saved_merges = defaultdict(lambda: defaultdict(lambda: []))
        self.ids          = {}
        # Initialize 1-form clusters for each PoS.
        for pos, tab in model.pos_forms.items():
            self.ids[pos] = [[i] for i in range(len(tab))]
        self.param = defaultdict(lambda: defaultdict(lambda: []))
        # Compute all parameters for each lemma.
        for lem in self.lemmas:
            pos    = lem.split()[1]
            forms  = model.pos_forms[pos]
            n      = len(forms)
            matrix = np.full([n, n,], np.nan)
            merges = np.empty( (n,n), dtype=object )
            self.matrices[pos][lem]     = matrix
            self.saved_merges[pos][lem] = merges
            # Get the IDs of the forms
            form_id = list(range(len(forms)))
            # Initialize param
            self.param[pos][lem] = [[] for x in form_id]
            # The PoS has only one form.
            if len(forms) == 1:
                self.matrices[pos][lem][0,0]     = 0.0
                self.saved_merges[pos][lem][0,0] = []
                self.param[pos][lem][0]          = model.get_param(lem, 0)
            else:
                # Compute IG for the form pairs.
                for i, j in combinations(form_id, 2):
                    param1 = model.get_param(lem, i)
                    param2 = model.get_param(lem, j)
                    # Save form parameters:
                    self.param[pos][lem][i] = param1
                    self.param[pos][lem][j] = param2
                    # No parameters given for i or j (forms not seen with lemma)
                    if param1 == [] or param2 == []:
                        continue
                    # Compute information gain
                    param_merged = merge(param1, param2)
                    ig = compute_info_gain(param1, param2, param_merged)
                    # Add IG to the matrix
                    self.matrices[pos][lem][i,j] = self.matrices[pos][lem][j,i] = ig
                    # Save merges parameters
                    self.saved_merges[pos][lem][i,j] = self.saved_merges[pos][lem][j,i] = param_merged

    def merge_forms(self, pos, i, j):
        """
        Update lemma matrices after merge.
        """
        def _add_row_col_of_lists(matrix):
            """
            Add a row and a column to a matrix
            (initialized with nan).
            """
            n = len(matrix)
            new_row = np.full([n], np.nan)
            matrix  = np.vstack([matrix, new_row])
            new_col = np.full([n+1], np.nan)
            matrix  = np.column_stack((matrix, new_col))
            # New row and comlumns contains empty lists.
            i = len(matrix) -1
            for j in range(i):
                matrix[i,j] = matrix[j,i] = []
            return matrix

        # Get new class parameters
        for lem in self.matrices[pos]:
            new_param = self.saved_merges[pos][lem][i,j]
            # For this lemma, merged forms have not been seen together.
            if new_param == None:
                new_param = []
            # Remove former class parameters
            del self.param[pos][lem][i], self.param[pos][lem][j-1]
            self.matrices[pos][lem]     = matrix_del(self.matrices[pos][lem], i, j)
            self.saved_merges[pos][lem] = matrix_del(self.saved_merges[pos][lem], i, j)
            # Add new class
            self.param[pos][lem].append(new_param)
            self.matrices[pos][lem]     = add_row_col(self.matrices[pos][lem])
            self.saved_merges[pos][lem] = _add_row_col_of_lists(self.saved_merges[pos][lem])
            # Compute merge parameters and IGs for the new class
            l = len(self.matrices[pos][lem]) - 1
            for k in range(l):
                param1 = self.param[pos][lem][k]
                param2 = self.param[pos][lem][l]
                # No parameters given for k or l (word not seen in data)
                if param1 == [] or param2 == []:
                    continue
                # Compute information gain
                param_merged = merge(param1, param2)
                ig = compute_info_gain(param1, param2, param_merged)
                # Add IG to the matrix
                self.matrices[pos][lem][k,l] = self.matrices[pos][lem][l,k] = ig
                # Save merges parameters
                self.saved_merges[pos][lem][k,l] = self.saved_merges[pos][lem][l,k] = param_merged

    def compute_new_igs(self, pos):
        """
        compute new IG sum for the new IG vector.
        """
        n = len(self.ids[pos])
        vec_ig = np.full([n,], np.nan)
        for m in self.matrices[pos].values():
            new_vec = m[-1]
            vec_ig  = matrix_add(new_vec, vec_ig)
        return vec_ig            


class ClustersPos():
    """
    Compute and store the IGs for all word pair merges
    (at PoS level).

    :type ig_pos: dict(numpy.ndarray(list(float)))
    """
    def __init__(self, tables_ig):
        self.ig_pos = {}
        for pos, forms in tables_ig.ids.items():
            n = len(forms)
            self.ig_pos[pos] = np.full([n, n,], np.nan)
            for igs in tables_ig.matrices[pos].values():
                # Increment ig_pos (implicitly consider nans as zeros)
                self.ig_pos[pos] = matrix_add(self.ig_pos[pos], igs)

    def cluster_by_pos(self, pos, ig_min):
        """
        Unlexicalized clustering.
        """
        while True:
            # Get the argmax in the IG matrix
            i, j = argmax(self.ig_pos[pos])
            # Get form list
            ids  = tables_ig.ids[pos]
            # End the search when max < min IG or when 
            # there is only one class left.
            if i == j == None or self.ig_pos[pos][i][j] < ig_min or len(ids) == 1:
                break
            # i must be lower than j
            if i > j:
                i, j = j, i
            # Update classes in ids
            id_f1  = tables_ig.ids[pos][i]
            id_f2  = tables_ig.ids[pos][j]
            new_id = id_f1 + id_f2
            del tables_ig.ids[pos][i], tables_ig.ids[pos][j-1]
            tables_ig.ids[pos].append(new_id)
            # Remove merged classes
            # from lemma IG matrices
            tables_ig.merge_forms(pos, i, j)
            # and from ig_pos
            self.ig_pos[pos] = matrix_del(self.ig_pos[pos], i, j)
            # Add new class to ig_pos
            self.ig_pos[pos] = add_row_col(self.ig_pos[pos])
            # Compute new class igs
            vec_new_igs = tables_ig.compute_new_igs(pos)
            # update ig_pos
            for k in range(len(self.ig_pos[pos])):
                self.ig_pos[pos][-1,k] = self.ig_pos[pos][k,-1] = vec_new_igs[k]
                    
        return tables_ig.ids[pos]


def add_row_col(matrix):
    """
    Add a row and a column to a matrix
    (initialized with nan).
    """
    n = len(matrix)
    new_row = np.full([n], np.nan)
    matrix  = np.vstack([matrix, new_row])
    new_col = np.full([n+1], np.nan)
    matrix  = np.column_stack((matrix, new_col))
    return matrix

def matrix_add(m1, m2):
    """
    Make matrix addition considering nans as zeros.
    """
    m1 = np.ma.masked_array(
        np.nan_to_num(m1),
        mask=np.isnan(m1) & np.isnan(m2)
        )
    m2 = np.ma.masked_array(np.nan_to_num(m2),
                            mask=m1.mask)
    return (m1 + m2).filled(np.nan)

def matrix_del(matrix, i, j):
    """
    Delete a row and a column from a matrix.
    """
    matrix = np.delete(matrix, (i), axis=0)
    matrix = np.delete(matrix, (j-1), axis=0)
    matrix = np.delete(matrix, (i), axis=1)
    matrix = np.delete(matrix, (j-1), axis=1)
    return matrix

def argmax(matrix):
    """
    Get both coordinates of the argmax in a matrix.
    Returns x and y coordinates as int (both are
    None if the matrix contains only 'nan')
    """
    # Get the argmax in the IG matrix
    try:
        flat_index = np.nanargmax(matrix)
    # The matrix contains only 'nan'
    except ValueError:
        return None, None
    # Get argmax as 2 int
    l = len(matrix)
    i = int(flat_index / l)
    j = flat_index % l
    return i, j

def norm_cnts(cnts):
    """
    Normalize counts stored in a dictionary
    """
    norm = sum(cnts.values())
    return {k: v/norm for (k, v) in cnts.items()}

def compute_loc_entropy(probs):
    """
    Take a probability distribution and return
    its entropy.
    """
    len_p = len(probs)
    if len_p == 1:
        return 0.0
    norm = ( math.log( len_p, 2 ) )
    return sum([(-p * math.log(p, 2)) for p in probs]) / norm

def merge(param1, param2):
    """
    Merge two word forms to obtain f' and return p(f'),
    H(e|f') and c(e|f').
    """
    # p(f')
    prob_new_f  = param1[0] + param2[0]
    # c(e|f')
    distrib_new =  param1[2] + param2[2]
    # H(e|f')
    distrib_new_norm = norm_cnts(distrib_new)
    entropy_new      = compute_loc_entropy(distrib_new_norm.values())
    return [prob_new_f, entropy_new, distrib_new]
    
def compute_info_gain(param1, param2, merged):
    """
    Return information gain from merging two forms.
    """
    p1 = param1[0] 
    h1 = param1[1]
    p2 = param2[0] 
    h2 = param2[1]
    h_combination = p1*h1 + p2*h2
    h_fusion = merged[0] * merged[1]
    return h_combination - h_fusion


parser = argparse.ArgumentParser(description = \
"""
Bilingual morph normalizer.
Normalize the source language with respect to the
target language by merging source words that have
a similar distribution over target words p(e|f)
using an entropy-based criterion.
In the source file, words are represented as a
sequence of one lemma, one PoS and tags, separated
by space. Each word representation is separated
by tabs. Ex.: je Pron Sg Ps1 - [TAB] normaliser Vb Sg Ps1 Pres
""")
parser.add_argument('-s', '--src', dest='src', nargs='?', type=argparse.FileType('r'),
                    help="source file in the language to normalize (space between attributes, tabs between words)")
parser.add_argument('-t', '--trg', dest='trg', nargs='?', type=argparse.FileType('r'),
                    help="target file")
parser.add_argument('-a', '--ali', dest='ali', nargs='?', type=argparse.FileType('r'),
                    help="src-to-trg alignment file")
parser.add_argument('-n', '-normal-model', dest='norm', type=str, default='norm_model.pkl',
                    help="pickle file where normalization model is dumped")
parser.add_argument('-r', '--rev', dest='rev', action='store_true',
                    help="reverse alignments (change src-to-trg into trg-to-src alignments)")
parser.add_argument('-l', '-lem-min', dest='lem_min', type=int, default=100,
                    help="minimum lemma frequency for lexicalized normalization (default: %(default)s)")
parser.add_argument('-f', '-form-min', dest='form_min', type=int, default=10,
                    help="minimum form frequency for lexicalized normalization (default: %(default)s)")
parser.add_argument('-m', '-ig-min', dest='ig_min', type=float, default=0.0,
                    help="minimum information gain for form merges (default: %(default)s)")
parser.add_argument('--pos', dest='pos', type=str, default='',
                    help="restrict normalization to a set of PoS (PoS names linked with '-')")
parser.add_argument('-use-mean', dest='use_mean', action='store_true',
                    help="use mean of lemma-level merges to estimate class merges")
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help="display learnt clusters on stdout")
args = parser.parse_args()


sys.stderr.write("* Loading data and counting\n")
model = Distributions(args.src, args.trg, args.ali, args.rev, args.lem_min, args.form_min, args.pos)

# Start normalizing
sys.stderr.write("* Learning optimal normalization (over {} lemmas)\n".format(len(model.param)))
# Get IG tables for lemmas
tables_ig = IgMatrix(model)

# Start clustering
clustered_pos = {}
if args.use_mean:
    from pos_mean import ClustersPosMean
    tables_pos = ClustersPosMean(model.pos_forms)
    for lem in model.param:
        pos = lem.split()[1]
        forms = model.pos_forms[pos]
        # Create IG matrix for the lemma
        tables_ig_lem    = tables_ig.matrices[pos][lem]
        tables_ig_merges = tables_ig.saved_merges[pos][lem]
        tables_pos.update_freq(pos, tables_ig_lem, tables_ig_merges)
    # Normalize frequency matrix for PoS
    tables_pos.normalize_freq()
    for pos in model.pos_forms:
        known_forms = [0 for x in model.pos_forms[pos]]
        forms = [[i] for i, j in enumerate(model.pos_forms[pos])]
        clustered_pos[pos] = tables_pos.cluster_by_pos_mean(pos, forms, args.ig_min, known_forms)

else:
    # Store fusions at PoS level
    tables_pos = ClustersPos(tables_ig)
    for pos in tables_pos.ig_pos:
        clustered_pos[pos] = tables_pos.cluster_by_pos(pos, args.ig_min)

if args.verbose:
    for pos in clustered_pos:
        print("===== PoS:", pos, "=====")
        for n, clusters in enumerate(clustered_pos[pos]):
            print("{}:".format(n), ' | '.join([model.pos_forms[pos][i] for i in clusters]))
        print()

# Output normalization model
sys.stderr.write("* Outputting normalization model\n")
# Format model
for pos, clusters in clustered_pos.items():
    forms = []
    for cluster in clusters:
        forms.append([model.pos_forms[pos][i] for i in cluster])
    clustered_pos[pos] = list(forms)
# Get sorted lemmas by frequency
lem_freq = [l[0] for l in model.unig_lem]
normal_model = [clustered_pos, lem_freq]
pickle.dump( normal_model, open( args.norm, 'wb'), protocol=2 )
