import math
import os
import sys
import time
from collections import defaultdict
from itertools import product
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr

from fileio import *
from project_globals import *
from similarities_mm import *

similarity_function_name = ""

# -- main -------------------------------------------------------------------#
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s <similarity measure>" % sys.argv[0])
    similarity_function_name = sys.argv[1]
    if not similarity_function_name in similarity_measures.keys():
        sys.exit(
            f"Invalid similarity measure: {sys.argv[1]}\n"
            f"Available measures: {similarity_measures.keys()}"
        )
    similarity_function = similarity_measures[sys.argv[1]]

    outputdir = "output" + time.strftime("-%m-%d-%H-%M/")
    try:
        os.mkdir(outputdir)
    except FileExistsError:
        print(f"{outputdir} exists")

    qid_similarity_list_dict = defaultdict(list)
    qid_ap_list_dict = defaultdict(list)

    ############################################################
    # Read query and AP files
    ideal_term_weight_dict = read_query_term_weight_file(ideal_query_path)
    ideal_ap_dict = read_qid_ap_file(ideal_ap_file_path)

    for coll in colls:
        for exp_method, num_docs, num_terms in product(
            expansion_methods, num_top_docs, num_exp_terms
        ):
            runid = "-".join(
                [exp_method, str(num_terms), str(num_docs), method_params[exp_method]]
            )
            exp_query_file = (
                "/".join([expanded_query_path, exp_method, "weights", runid])
                + ".term_weights"
            )
            ap_file = "/".join([expanded_query_path, exp_method, "aps", runid]) + ".ap"

            # print(exp_query_file, ap_file)
            # continue

            with open(outputdir + "-".join(["matches", coll, runid]), "w") as f:
                log_sim.info(f"{runid}")
                qid_term_weight_dict = read_query_term_weight_file(exp_query_file)
                ap_dict = read_qid_ap_file(ap_file)

                ############################################################
                # Compute similarity list and AP list for each query
                for qid in ideal_term_weight_dict.keys():
                    if qid not in ideal_ap_dict:
                        continue

                    ideal_q = prepare_query(ideal_term_weight_dict[qid], math.inf)
                    exp_q = prepare_query(qid_term_weight_dict[qid], num_terms)

                    log_sim.debug(f"{ideal_q}")
                    log_sim.debug(f"{exp_q}")

                    sim = similarity_function(qid, ideal_q, exp_q, f)
                    qid_similarity_list_dict[qid].append(sim)
                    qid_ap_list_dict[qid].append(ap_dict[qid])

        ############################################################
        # Compute correlation for each query
        with open(
            outputdir + "-".join(["paired-lists", similarity_function_name, coll]), "w"
        ) as f, open(
            outputdir
            + "-".join(["query-wise-correlation", similarity_function_name, coll]),
            "w",
        ) as g:

            sum_of_corr_pear = 0
            sum_of_corr_kend = 0
            sum_of_corr_spear = 0
            non_nan_count_pearson = 0
            non_nan_count_kendall = 0
            non_nan_count_spearman = 0

            for qid in qid_similarity_list_dict.keys():
                similarity_list = qid_similarity_list_dict[qid]
                ap_list = qid_ap_list_dict[qid]

                assert len(similarity_list) == num_correlation_pairs
                assert len(ap_list) == num_correlation_pairs
                print(qid, *similarity_list, sep=" ", file=f)
                print(qid, *ap_list, sep=" ", file=f)

                pearson_corr, _ = pearsonr(ap_list, similarity_list)
                kendall_corr, _ = kendalltau(ap_list, similarity_list)
                spearman_corr, _ = spearmanr(ap_list, similarity_list)
                print(qid, pearson_corr, kendall_corr, spearman_corr, file=g)

                if math.isnan(pearson_corr) != True:
                    sum_of_corr_pear += pearson_corr
                    non_nan_count_pearson += 1
                if math.isnan(kendall_corr) != True:
                    sum_of_corr_kend += kendall_corr
                    non_nan_count_kendall += 1
                if math.isnan(spearman_corr) != True:
                    sum_of_corr_spear += spearman_corr
                    non_nan_count_spearman += 1

        with open(
            outputdir + "-".join(["summary", similarity_function_name, coll]), "w"
        ) as f:
            print(f"Correlation over {num_correlation_pairs} pairs", file=f)
            print(
                "avg corr pearson: ", sum_of_corr_pear / non_nan_count_pearson, file=f
            )
            print(
                "# non nan point for how many queries : ", non_nan_count_pearson, file=f
            )
            print(
                "avg corr kendall: ", sum_of_corr_kend / non_nan_count_kendall, file=f
            )
            print(
                "# non nan point for how many queries (kendall, file=f) : ",
                non_nan_count_kendall,
                file=f,
            )
            print(
                "avg corr spearman: ",
                sum_of_corr_spear / non_nan_count_spearman,
                file=f,
            )
            print(
                "# non nan point for how many queries (spearman, file=f) : ",
                non_nan_count_spearman,
                file=f,
            )
