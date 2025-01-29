import os
import pytrec_eval
from collections import defaultdict
import argparse

from definitions import ROOT_DIR
from utils import store_qrel

parser = argparse.ArgumentParser(
    description="Intersect a run file and a qrel file and return a qrel file containing only those documents that occur in the run file."
)
parser.add_argument("run_path", type=str, help="Path to the run file")
parser.add_argument("qrel_path", type=str, help="Path to the qrel file")
parser.add_argument(
    "qrel_output_path", type=str, help="Path to save the output qrel file"
)
args = parser.parse_args()

run_path = os.path.join(ROOT_DIR, args.run_path)
qrel_path = os.path.join(ROOT_DIR, args.qrel_path)
qrel_output_path = os.path.join(ROOT_DIR, args.qrel_output_path)

qrel = pytrec_eval.parse_qrel(open(qrel_path, "r"))
run = pytrec_eval.parse_run(open(run_path, "r"))

qrel_final = defaultdict(dict)

# Build qrel_final by intersecting qrel and run
for qid, relevances in qrel.items():
    for docid, rel in relevances.items():
        if docid in run[qid]:
            qrel_final[qid][docid] = rel

store_qrel(qrel_final, qrel_output_path)
