import os
import argparse
import re
from pathlib import Path

from searcher import SearchAndEval
from utils import parse_queries, store_run, store_ap, store_mean_ap
from definitions import TREC_INDEX_DIR_PATH, STOPWORDS_FILE_PATH, TREC_QREL_FILE_PATH


"""
Expected input: term_weights file
Expected output: run file, ap file in corresponding 'runs' and 'aps' dir beside parent dir of term_weights file
"""

parser = argparse.ArgumentParser()
parser.add_argument("weight_file", help="Query Weight File", type=str)
args = parser.parse_args()
runid = re.sub(r"\.term_weights", "", os.path.basename(args.weight_file))

dirname = Path(args.weight_file).absolute().parent.parent

store_run_file = os.path.join(dirname, "runs", f"{runid}.run")
store_ap_file = os.path.join(dirname, "aps", f"{runid}.ap")

searcher = SearchAndEval(TREC_INDEX_DIR_PATH, STOPWORDS_FILE_PATH, TREC_QREL_FILE_PATH)
print("Index initialized")

query_vectors = parse_queries(args.weight_file)

# ## For expanded queries, extract num terms from the file name
# num_terms = int(
#     os.path.basename(args.weight_file).split("-")[1]
# )  # for trimming as per given number of terms in expanded query files

# num_terms = 1000
# num_terms = 200
num_terms = None

for qid, query_vector in query_vectors.items():
    query_vector.remove_non_positive_weights()
    query_vector.sort_by_stat()
    if num_terms is not None:
        query_vector.trim(num_terms)  # trim upto top num_terms terms
    print(f"Running Query {qid}", end=" ")
    ap, run = searcher.computeAP_and_run(qid, query_vector)
    ap_dict = dict()
    ap_dict[qid] = ap
    store_run(run, store_run_file, runid, append=True)
    if ap is not None:
        store_ap(ap_dict, store_ap_file, append=True)
    print(ap)
store_mean_ap(store_ap_file)
