import pytrec_eval
from collections import defaultdict
import argparse

from utils import store_qrel

parser = argparse.ArgumentParser()
parser.add_argument("run_path", type=str, help="Path to the run file")
parser.add_argument("qrel_path", type=str, help="Path to the qrel file")
parser.add_argument(
    "qrel_output_path", type=str, help="Path to save the output qrel file"
)
args = parser.parse_args()

qrel = pytrec_eval.parse_qrel(open(args.qrel_path, "r"))
run = pytrec_eval.parse_run(open(args.run_path, "r"))
qrel_final = defaultdict(dict)

# METHOD 1. Go through qrel file and see if it occurs in the run file
# If it does, add it to the new qrel list with its corresponding qrel value
for qid, relevances in qrel.items():
    for docid, rel in relevances.items():
        if docid in run[qid]:
            qrel_final[qid][docid] = rel

# METHOD 2. Go through the run file and assign each document's relevance based on qrel value
# for qid, doc_scores in run.items():
#     for docid in doc_scores.keys():
#         qrel_final[qid][docid] = qrel[qid].get(docid, 0)

store_qrel(qrel_final, args.qrel_output_path)
