# Each similarity function takes 3 arguments: ideal_query, expanded_query, logfile
# Input queries ≡ list of (term, weight, rank) tuples, sorted alphabetically by term

import logging
import math
import numpy as np
from numpy import linalg as la
import csv

from fileio import *

logging.basicConfig(
    filename="similarities.log",
    filemode="w",  # 'a' (append) by default
    level=logging.INFO,  # increasing order: debug, info, warning [default], error, critical
    format="%(message)s",
)  # use %(name)s, %(process)d, %(asctime)s, %(levelname)s' as required
log_sim = logging.getLogger("similarities")

num_rel = dict()
with open("./num_rel_docs") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        num_rel[int(row[0])] = int(row[1])


def jaccard(qid, ideal_query, expanded_query, logfile):
    intersection = []
    i, j, I, J = 0, 0, len(ideal_query), len(expanded_query)
    while i < I and j < J:
        if ideal_query[i][0] < expanded_query[j][0]:
            i += 1
        elif ideal_query[i][0] > expanded_query[j][0]:
            j += 1
        else:
            intersection.append(expanded_query[j] + ideal_query[i][1:])
            i, j = i + 1, j + 1

    print_matches(qid, intersection, logfile)
    print("\n", file=logfile)

    union = I + J - (l := len(intersection))
    print(f"union = {union} intersection = {l} " f"similarity {(sim := l/union)}")

    return sim


def cap(qid, ideal_query, expanded_query, logfile):
    intersection = []
    i, j, I, J = 0, 0, len(ideal_query), len(expanded_query)
    while i < I and j < J:
        if ideal_query[i][0] < expanded_query[j][0]:
            i += 1
        elif ideal_query[i][0] > expanded_query[j][0]:
            j += 1
        else:
            intersection.append(expanded_query[j] + ideal_query[i][1:])
            i, j = i + 1, j + 1

    print_matches(qid, intersection, logfile)
    print("\n", file=logfile)

    union = I + J - (l := len(intersection))
    print(f"union = {union} intersection = {l} " f"similarity {(sim := l)}")

    return sim


def jaccard_modified_1(ideal_query, expanded_query, logfile):
    pass


def jaccard_modified_2(ideal_query, expanded_query, logfile):
    pass


def l1_similarity(qid, ideal_query, expanded_query, logfile):
    intersection = []
    l1_expq = sum([abs(x[1]) for x in expanded_query])
    sim = 0.0
    i, j, I, J = 0, 0, len(ideal_query), len(expanded_query)
    while i < I and j < J:
        if ideal_query[i][0] < expanded_query[j][0]:
            i += 1
        elif ideal_query[i][0] > expanded_query[j][0]:
            j += 1
        else:
            intersection.append(expanded_query[j] + ideal_query[i][1:])
            sim += ideal_query[i][1] * expanded_query[j][1]
            i, j = i + 1, j + 1

    print_matches(qid, intersection, logfile)
    print("\n", file=logfile)

    return sim / l1_expq


def dot_product(qid, ideal_query, expanded_query, logfile):
    intersection = []
    sim = 0.0
    i, j, I, J = 0, 0, len(ideal_query), len(expanded_query)
    while i < I and j < J:
        if ideal_query[i][0] < expanded_query[j][0]:
            i += 1
        elif ideal_query[i][0] > expanded_query[j][0]:
            j += 1
        else:
            intersection.append(expanded_query[j] + ideal_query[i][1:])
            sim += ideal_query[i][1] * expanded_query[j][1]
            i, j = i + 1, j + 1

    print_matches(qid, intersection, logfile)
    print("\n", file=logfile)

    return sim


def l2_similarity(qid, ideal_query, expanded_query, logfile):
    intersection = []
    l2_expq = la.norm(np.array([x[1] for x in expanded_query]))
    sim = 0.0
    i, j, I, J = 0, 0, len(ideal_query), len(expanded_query)
    while i < I and j < J:
        if ideal_query[i][0] < expanded_query[j][0]:
            i += 1
        elif ideal_query[i][0] > expanded_query[j][0]:
            j += 1
        else:
            intersection.append(expanded_query[j] + ideal_query[i][1:])
            sim += ideal_query[i][1] * expanded_query[j][1]
            i, j = i + 1, j + 1

    print_matches(qid, intersection, logfile)
    print("\n", file=logfile)

    # return (sim / l2_expq) / math.log(num_rel[qid])
    return sim / l2_expq


def ndcg(ideal_query, expanded_query, logfile):
    pass


def ndcg_modified_1(ideal_query, expanded_query, logfile):
    pass


def ndcg_modified_2(qid, ideal_query, expanded_query, logfile):
    intersection = []
    sim = 0.0
    i, j, I, J = 0, 0, len(ideal_query), len(expanded_query)
    while i < I and j < J:
        if ideal_query[i][0] < expanded_query[j][0]:
            i += 1
        elif ideal_query[i][0] > expanded_query[j][0]:
            j += 1
        else:
            intersection.append(expanded_query[j] + ideal_query[i][1:])
            sim += (ideal_query[i][1] * 1000) / (1000 + (j + 1))
            i, j = i + 1, j + 1

    ideal_sim = 0.0
    for i in range(len(expanded_query)):
        ideal_sim += (ideal_query[i][1] * 1000) / (1000 + (i + 1))

    print_matches(qid, intersection, logfile)
    print("\n", file=logfile)

    return sim / ideal_sim


# ---------------------------------------------------------------------------#

similarity_measures = {
    "j": jaccard,
    "j1": jaccard_modified_1,
    "j2": jaccard_modified_2,
    "l1": l1_similarity,
    "l2": l2_similarity,
    "n": ndcg,
    "n1": ndcg_modified_1,
    "n2": ndcg_modified_2,
    "dp": dot_product,
    "cap": cap,
}
