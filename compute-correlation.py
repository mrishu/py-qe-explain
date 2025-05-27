import os
from itertools import product
from collections import defaultdict
from scipy.stats import pearsonr, kendalltau, spearmanr
import csv
import math

from definitions import ROOT_DIR
from utils import parse_queries, parse_ap
from similarities import Similarities

# GLOBALS #
###########################################################
ideal_q_runid = "ideal_query"
similarity_names = ["j", "l1", "l2", "n2"]

output_dir = os.path.join(
    ROOT_DIR, "correlation-computations", "trec678", f"{ideal_q_runid}"
)
if os.path.exists(output_dir):
    inp = input("Output directory already exists. Overwrite? (Y/N): ")
    if (inp == "y") or (inp == "Y"):
        os.rmdir(output_dir)
    else:
        exit(1)
os.makedirs(output_dir)

# ideal query path
ideal_query_path = os.path.join(ROOT_DIR, "ideal-queries", "trec678")
ideal_query_weight_file = os.path.join(
    ideal_query_path, "weights", ideal_q_runid + ".term_weights"
)

# expanded_query_path
expanded_query_path = os.path.join(ROOT_DIR, "expanded-queries", "trec678")

# parameter grid
expansion_methods = ["rm3", "ceqe", "loglogistic", "spl"]
num_exp_terms = [15, 25, 35, 45, 55]
num_top_docs = [10, 20, 30, 40]
method_params = {
    "rm3": "0.6",
    "ceqe": ".6",
    "loglogistic": "2",
    "spl": "8",
}

###########################################################

num_correlation_pairs = len(expansion_methods) * len(num_top_docs) * len(num_exp_terms)

# Store each ideal query in a Similarity class
ideal_queries = parse_queries(ideal_query_weight_file)
sim_obj = dict()
for qid, ideal_qvec in ideal_queries.items():
    print("Reading ideal query number:", qid)
    sim_obj[qid] = Similarities(qid, ideal_qvec)

sim_dict = dict()
# sim_dict: sim_name -> (qid -> list of sim scores of each expanded query variant)
for sim_name in similarity_names:
    sim_dict[sim_name] = defaultdict(list)
ap_dict = defaultdict(
    list
)  # ap_dict: qid -> list of APs of each expanded query variant

for exp_method, num_docs, num_terms in product(
    expansion_methods, num_top_docs, num_exp_terms
):
    exp_runid = f"{exp_method}-{num_terms}-{num_docs}-{method_params[exp_method]}"
    exp_query_file = (
        os.path.join(expanded_query_path, exp_method, "weights", exp_runid)
        + ".term_weights"
    )
    exp_ap_file = (
        os.path.join(expanded_query_path, exp_method, "aps", exp_runid) + ".ap"
    )

    exp_queries = parse_queries(exp_query_file)
    exp_query_aps = parse_ap(exp_ap_file)

    matches_file_name = f"matches-{exp_runid}.csv"
    print("Writing", matches_file_name)

    for qid, qvec in exp_queries.items():
        if qid not in exp_query_aps or qid not in sim_obj:
            continue
        qvec.sort_by_stat()
        qvec.trim(num_terms)
        sim_obj[qid].change_expanded_query(qvec)
        sim_obj[qid].print_matches(os.path.join(output_dir, matches_file_name))
        ap_dict[qid].append(exp_query_aps[qid])
        for sim_name in similarity_names:
            sim_dict[sim_name][qid].append(sim_obj[qid].compute_similarity(sim_name))

for sim_name in similarity_names:
    paired_list_file_name = f"paired-lists-{sim_name}.csv"
    query_wise_corr_file_name = f"paired-lists-{sim_name}.csv"
    print("Writing", paired_list_file_name)
    paired_list_file = open(os.path.join(output_dir, paired_list_file_name))
    print("Writing", f"query-wise-correlation-{sim_name}.csv")
    query_wise_corr_file = open(os.path.join(output_dir, query_wise_corr_file_name))

    pearson_corr_list = []
    kendall_corr_list = []
    spearman_corr_list = []
    paired_list_writer = csv.writer(paired_list_file, delimiter="\t")
    query_wise_corr_writer = csv.writer(query_wise_corr_file, delimiter="\t")

    for qid in ap_dict:
        ap_list = ap_dict[qid]
        similarity_list = sim_dict[sim_name][qid]

        pearson_corr, _ = pearsonr(ap_list, similarity_list)
        kendall_corr, _ = kendalltau(ap_list, similarity_list)
        spearman_corr, _ = spearmanr(ap_list, similarity_list)

        if not math.isnan(pearson_corr):
            pearson_corr_list.append(pearson_corr)
        if not math.isnan(kendall_corr):
            kendall_corr_list.append(kendall_corr)
        if not math.isnan(spearman_corr):
            spearman_corr_list.append(spearman_corr)

        paired_list_writer.writerow([qid, *similarity_list])
        paired_list_writer.writerow([qid, *ap_list])
        query_wise_corr_writer.writerow(
            [qid, pearson_corr, kendall_corr, spearman_corr]
        )

    paired_list_file.close()
    query_wise_corr_file.close()

    summary_file_name = f"summary-{sim_name}.txt"
    print("Writing", summary_file_name)

    with open(os.path.join(output_dir, summary_file_name), "w") as summary_file:
        print(f"Correlation over {num_correlation_pairs} pairs", file=summary_file)
        print(
            "avg corr pearson: ",
            sum(pearson_corr_list) / max(len(pearson_corr_list), 1),
            file=summary_file,
        )
        print(
            "# non nan point for how many queries (pearson) : ",
            len(pearson_corr_list),
            file=summary_file,
        )
        print(
            "avg corr kendall: ",
            sum(kendall_corr_list) / max(len(kendall_corr_list), 1),
            file=summary_file,
        )
        print(
            "# non nan point for how many queries (kendall) : ",
            len(kendall_corr_list),
            file=summary_file,
        )
        print(
            "avg corr spearman: ",
            sum(spearman_corr_list) / max(len(spearman_corr_list), 1),
            file=summary_file,
        )
        print(
            "# non nan point for how many queries (spearman) : ",
            len(spearman_corr_list),
            file=summary_file,
        )
