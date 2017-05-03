#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import operator
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

class ClustersPosMean():
    """
    Store the IGs for all word pair merges.

    :type ig_pos: dict(numpy.ndarray(list(float)))
    :type freq_cooc: dict(numpy.ndarray(list(float)))
    """
    def __init__(self, pos_forms):
        self.pos_forms = pos_forms.copy()
        self.ig_pos    = {}
        self.freq_cooc = {}
        for pos, forms in self.pos_forms.items():
            n = len(forms)
            self.ig_pos[pos] = arrays_in_matrix(n)
            self.freq_cooc[pos] = arrays_in_matrix(n)

    def update_freq(self, pos, ig_matrix, merges):
        """
        After the lemma has been clustered, add IGs
        for all possible merges.
        """
        if len(ig_matrix) > 1:
            n = len(ig_matrix)
            for i, j in combinations(list(range(n)), 2):
                # Increment all computed IGs for the lemma.
                ig = ig_matrix[i,j]
                if np.isnan(ig):
                    continue
                self.ig_pos[pos][i,j].append(ig)
                self.ig_pos[pos][j,i].append(ig)
                # Collect form cooccurrences probability
                prob = merges[i,j][0]
                self.freq_cooc[pos][i,j].append(prob)
                self.freq_cooc[pos][j,i].append(prob)

    def normalize_freq(self):
        """
        After gathering counts over frequent lemmas,
        normalize them.
        
        :type freq_norm: dict(numpy.ndarray(float))
        """
        self.freq_norm = {}
        for pos in self.ig_pos.keys():
            n = len(self.ig_pos[pos])
            self.freq_norm[pos] = np.full([n, n,],np.nan)
            for i, j in combinations(list(range(n)), 2):
                igs = self.ig_pos[pos][i,j]
                # The forms have never been seen together.
                if igs == []:
                    continue
                probs    = self.freq_cooc[pos][i,j]
                ig_norm  = [igs[k] for k in range(len(probs))]
                avrg_ig  = sum(ig_norm)
                # print("summed:", avrg_ig)
                self.freq_norm[pos][i,j] = self.freq_norm[pos][j,i] = avrg_ig               
        del self.ig_pos, self.freq_cooc

    def cluster_by_pos_mean(self, pos, ids, ig_min, known_words):
        """
        Unlexicalized clustering.
        """
        # Build the IG matrix, given the clusters
        cluster_ig = self.init_ig_matrix(pos, ids, known_words)
        while True:
            # Get the argmax in the IG matrix
            i, j = argmax(cluster_ig)
            # End the search when max < min IG or when 
            # there is only one class left.
            if i == j == None or cluster_ig[i][j] < ig_min or len(ids) == 1:
                break

            # Merge classes
            id_f1 = ids[i]
            id_f2 = ids[j]
            new_id = id_f1 + id_f2
            if i > j:
                i, j = j, i
            # One of merged classes was a known word, so is the new class.
            new_known = 0
            if ids[i] or ids[j]:
                new_known = 1
            del ids[i], ids[j-1]
            cluster_ig = matrix_del(cluster_ig, i, j)

            # Add new class
            ids.append(new_id)
            known_words.append(new_known)
            cluster_ig = add_row_col(cluster_ig)

            # Update new row and column
            j = len(cluster_ig) - 1
            for i in range(j):
                cluster_ig[i,j] = cluster_ig[j,i] = self.compute_delex_value(i, j, ids, known_words, pos)
        return ids

    def init_ig_matrix(self, pos, ids, known_words):
        """
        Make a matrix of delexicalized IG given a set of clusters.
        """
        n = len(ids)
        # Initialize values with minimal IG possible.
        matrix = np.full((n,n), -1.0)
        for i, j in combinations(list(range(n)), 2):
            matrix[i,j] = matrix[j,i] = self.compute_delex_value(i, j, ids, known_words, pos)
        return matrix

    def compute_delex_value(self, i, j, ids, known_words, pos):
        """
        Compute delexicalied values in the averaged IG matrix.
        """
        if known_words[i] and known_words[j]:
            return np.nan
        else:
            c1 = ids[i]
            c2 = ids[j]
            avrg_ig = []
            for f1 in c1:
                for f2 in c2:
                    avrg_ig.append(self.freq_norm[pos][f1][f2])
            # If one of the values of 'avrg_ig' is 'nan',
            # the function returns 'nan'.
            return sum(avrg_ig) / len(avrg_ig)


def arrays_in_matrix(n):
    """
    Return an n*n dimensional matrix containing lists.
    """
    aim = np.empty((n,n), dtype=object)
    for i, j in combinations(list(range(n)), 2):
        aim[i,j] = aim[j,i] = []
    return aim

def add_row_col(matrix):
    """
    Add a row and a column to a matrix
    (initialized with nan).
    """
    n = len(matrix)
    new_row = np.full([n], np.nan)
    matrix = np.vstack([matrix, new_row])
    new_col = np.full([n+1], np.nan)
    matrix = np.column_stack((matrix, new_col))
    return matrix

def matrix_del(matrix, i, j):
    """
    Delete a row and a column from a matrix.
    i and j are int, such that i < j.
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
    prob_new_f = param1[0] + param2[0]
    # c(e|f')
    distrib_new =  param1[2] + param2[2]
    # H(e|f')
    distrib_new_norm = norm_cnts(distrib_new)
    entropy_new = compute_loc_entropy(distrib_new_norm.values())
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


