from collections import defaultdict
import math

from project_globals import *


def defaultdict_float():
    return defaultdict(float)


def read_query_term_weight_file(file_path):
    # NOTE: need to pickle these dicts => a lambda will not do
    # see https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
    qid_term_weight_dict = defaultdict(defaultdict_float)

    with open(file_path) as term_weight_file:
        for line in term_weight_file:
            qid, term, weight = line.split()
            weight = weight.strip("\n")
            qid_term_weight_dict[qid][term] += float(weight)

    return qid_term_weight_dict


def read_qid_ap_file(file_path):
    qid_ap = defaultdict(float)

    with open(file_path) as qid_ap_file:
        for line in qid_ap_file:
            _, qid, ap_value = line.split()
            ap_value = ap_value.strip("\n")
            qid_ap[qid] = float(ap_value)

    return qid_ap


def prepare_query(query, length):
    """
    Input:
    - query ≡ dict (keys = terms, values = weights) for ideal/expanded query
    - length ≡ desired query length
    Output: list of (term, weight, rank) tuples, sorted alphabetically by term"""
    # sort by weight
    q = sorted(query.items(), key=lambda x: x[1], reverse=True)
    # select terms if necessary
    if length != math.inf:
        q = q[:length]
    # compute rank according to weight
    q = [(t[0], t[1], i + 1) for i, t in enumerate(q)]
    # sort alphabetically by term
    q.sort(key=lambda x: x[0])

    return q


def read_paired_lists(filename):
    """
    Input: name of file containing paired lists
    Returns: dictionary (key ≡ qid, value ≡ Pair(runid, similarity, ap, rank_s, rank_a))
    """
    paired_lists_dict = {}
    with open(filename) as f:  # with => don't need a f.close()
        while (s := f.readline()) and (a := f.readline()):
            # s, a are of the form: qid followed by 20 similarities / ap values
            s = s.strip().split()
            a = a.strip().split()
            assert len(s) == 81 and len(a) == 81 and s[0] == a[0]
            paired_lists_dict[a[0]] = [
                Pair(x[0], float(x[1][0]), float(x[1][1]), 0, 0)
                for x in enumerate(zip(s[1:], a[1:]))
            ]
    return paired_lists_dict


def print_matches(qid, M, f):
    """M is a list of (term, weight, rank, rank in ideal query) tuples"""
    for m in sorted(M, key=lambda x: x[2]):
        print(qid, *m, file=f)
