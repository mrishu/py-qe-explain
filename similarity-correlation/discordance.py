#!/usr/bin/env python3
# coding=utf-8

from dataclasses import dataclass
import math
from scipy.stats import pearsonr, kendalltau, spearmanr
from statistics import variance, covariance
import pickle
import sys

from fileio import *
import project_globals as myglobals
from similarities_mm import *


@dataclass
class QE_stats:
    qid: str
    mean_rank_sim: float = 0
    mean_rank_ap: float = 0
    max_sim: float = 0
    min_sim: float = 0
    max_ap: float = 0
    min_ap: float = 0
    sd_sim: float = 0
    sd_ap: float = 0


# ---------------------------------------------------------------------------#


def print_discordance(
    d, qid, ideal_queries, exp_queries, similarity_function, verbose=True
):
    print(
        f"{(runid := runids[d.runid])} "
        f"similarity = {d.similarity} ({d.rank_s}) "
        f"ap = {d.ap} ({d.rank_a})"
    )

    if not verbose:
        return

    r = runid.split("-")
    exp_method = r[0]
    num_terms = int(r[1])
    exp_query_file = (
        "/".join([myglobals.expanded_query_path, exp_method, "weights", runid])
        + ".term_weights"
    )

    # Read / lookup (if read earlier) expanded query for this runid
    if exp_query_file in exp_queries:
        qid_term_weight_dict = exp_queries[exp_query_file]
    else:
        qid_term_weight_dict = read_query_term_weight_file(exp_query_file)
        exp_queries[exp_query_file] = qid_term_weight_dict

    ideal_q = prepare_query(ideal_queries[qid], math.inf)
    exp_q = prepare_query(qid_term_weight_dict[qid], num_terms)

    sim = similarity_function(qid, ideal_q, exp_q, sys.stdout)
    response = input("Continue? ")
    if response.lower() != "y" and response.lower() != "yes":
        return


def examine_discordances_for_query(
    paired_lists_dict, ideal_queries, exp_queries, similarity_function
):
    while qid := input("Query no.? (return to quit): "):
        paired_list = paired_lists_dict[qid]

        # Sort by discordance
        paired_list.sort(key=lambda x: x.rank_s - x.rank_a)

        # Examine top discordances
        top_how_many = 5
        for d in paired_list[:top_how_many]:
            print_discordance(
                d, qid, ideal_queries, exp_queries, similarity_function, verbose=True
            )
        for d in paired_list[-top_how_many:]:
            print_discordance(
                d, qid, ideal_queries, exp_queries, similarity_function, verbose=True
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            f"Usage: {sys.argv[0]} <paired-lists file name>\n"
            f" (file name should be of the form paired-lists-j-trec678)"
        )

    filename = sys.argv[1].split("-")
    myglobals.coll = filename[-1]
    similarity_function = similarity_measures[filename[-2]]

    # Read ideal query file
    try:
        with open(myglobals.ideal_query_path + ".pickle", "rb") as f:
            ideal_queries_dict = pickle.load(f)
    except:
        print("Problem reading " + myglobals.ideal_query_path + ".pickle")
        ideal_queries_dict = read_query_term_weight_file(myglobals.ideal_query_path)
        with open(myglobals.ideal_query_path + ".pickle", "wb") as f:
            pickle.dump(ideal_queries_dict, f)

    # Read expanded query file
    try:
        with open("exp_queries_file.pickle", "rb") as f:
            exp_queries_dict = pickle.load(f)
    except:
        exp_queries_dict = {}

    # Read paired lists file
    try:
        with open(sys.argv[1] + ".pickle", "rb") as f:
            paired_lists_dict = pickle.load(f)
    except:
        print("Problem reading " + sys.argv[1] + ".pickle")
        paired_lists_dict = read_paired_lists(sys.argv[1])
        # Assign ranks
        for qid in paired_lists_dict:
            paired_list = paired_lists_dict[qid]
            paired_list.sort(key=lambda x: x.ap, reverse=True)
            for i, x in enumerate(paired_list):
                x.rank_a = i + 1
            paired_list.sort(key=lambda x: x.similarity, reverse=True)
            for i, x in enumerate(paired_list):
                x.rank_s = i + 1
        with open(sys.argv[1] + ".pickle", "wb") as f:
            pickle.dump(paired_lists_dict, f)

    examine_discordances_for_query(
        paired_lists_dict, ideal_queries_dict, exp_queries_dict, similarity_function
    )

    # Save similarity computations for next session
    with open("exp_queries_file.pickle", "wb") as f:
        pickle.dump(exp_queries_dict, f)

    # stats_dict = {}
    # for m in expansion_methods:
    #     stats_dict[m] = defaultdict(QE_stats)
    # for qid in paired_lists_dict:
    #     paired_list = paired_lists_dict[qid]
    #     for x in paired_list:
    #         r = runids[x.runid]
