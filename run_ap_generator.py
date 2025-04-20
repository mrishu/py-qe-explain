import os
import argparse
import re
from pathlib import Path

from searcher import SearchAndEval
from utils import parse_queries, store_run, store_ap
from definitions import TREC_INDEX_DIR_PATH, STOPWORDS_FILE_PATH, TREC_QREL_FILE_PATH


"""
Expected input: term_weights file stored in 'weights' folder
Expected output: run file, ap file in corresponding 'runs' and 'aps' folder beside the 'weights' folder
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
num_terms = int(
    os.path.basename(args.weight_file).split("-")[1]
)  # for trimming of longer query_vectors

for qid, query_vector in query_vectors.items():
    print(f"Running Query {qid}")
    query_vector.trim(num_terms)  # trim upto top num_terms terms
    ap, run = searcher.computeAP(qid, query_vector)
    ap_dict = dict()
    ap_dict[qid] = ap
    if ap:
        store_ap(ap_dict, store_ap_file, append=True)
        store_run(run, store_run_file, runid, append=True)
