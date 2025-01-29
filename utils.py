import csv
import os
from collections import defaultdict
from types import SimpleNamespace

from classes import QueryVector


def create_file_dir(file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


# Input:
# qrel: Dictionary containing the mapping "qid" -> ("docid" -> relevance)
# qrel_output_path: Path to save the qrel file
def store_qrel(qrel: dict[str, dict[str, int]], qrel_output_path: str, append=False):
    create_file_dir(qrel_output_path)
    with open(qrel_output_path, "w" if append else "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, relevances in qrel.items():
            for docid, rel in relevances.items():
                writer.writerow([qid, 0, docid, rel])


# Input:
# run: Dictionary containing the mapping "qid" -> ("docid" -> score)
# run_output_path: Path to save the run file
# runid: Run identifier
def store_run(
    run: dict[str, dict[str, float]], run_output_path: str, runid: str, append=False
):
    create_file_dir(run_output_path)
    with open(run_output_path, "a" if append else "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, scores in run.items():
            rank = 0
            for docid, score in scores.items():
                writer.writerow([qid, "Q0", docid, rank, score, runid])
                rank += 1


# Input:
# aps: Dictionary containing the mapping "qid" -> AP
# ap_file_path: Path to the AP file
def store_ap(aps: dict[str, float], ap_file_path: str, append=False):
    create_file_dir(ap_file_path)
    with open(ap_file_path, "a" if append else "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, ap in aps.items():
            writer.writerow(["map", qid, ap])


# Input:
# weight_file_path: Path to the weight file
# Output: Dictionary containing the mapping "qid" -> QueryVector
# Each QueryVector is (basically) a dictionary mapping "term" -> weight
def parse_queries(weight_file_path: str) -> dict[str, QueryVector]:
    with open(weight_file_path, "r") as f:
        query_vectors = defaultdict(QueryVector)
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            qid = row[0]
            term = row[1]
            weight = float(row[2])
            if term in query_vectors[qid]:
                query_vectors[qid][term].weight += weight
            else:
                query_vectors[qid][term] = SimpleNamespace(weight=weight)
    return query_vectors


# Input:
# ap_file_path: Path to the AP file
# Output: Dictionary containing the mapping "qid" -> AP
def parse_ap(ap_file_path: str) -> dict[str, float]:
    aps = dict()
    with open(ap_file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            qid = row[1]
            ap = float(row[2])
            aps[qid] = ap
    return aps
