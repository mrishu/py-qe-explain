import csv
import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Union

from classes import QueryVector


# Input:
# qrel: Dictionary containing the mapping dict("qid" -> dict("docid" -> relevance))
# qrel_output_path: Path to save the qrel file
def store_qrel(qrel: dict[str, dict[str, int]], qrel_output_path: str, append=False):
    os.makedirs(os.path.dirname(qrel_output_path), exist_ok=True)
    with open(qrel_output_path, "a" if append else "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, relevances in qrel.items():
            for docid, rel in relevances.items():
                writer.writerow([qid, 0, docid, rel])


# Input:
# run: Dictionary containing the mapping dict("qid" -> OrderedDict("docid" -> score))
# run_output_path: Path to save the run file
# runid: Run identifier
def store_run(
    run: dict[str, dict[str, float]], run_output_path: str, runid: str, append=False
):
    os.makedirs(os.path.dirname(run_output_path), exist_ok=True)
    with open(run_output_path, "a" if append else "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, scores in run.items():
            rank = 0
            for docid, score in scores.items():
                writer.writerow([qid, "Q0", docid, rank, score, runid])
                rank += 1


# Input:
# aps: Dictionary containing the mapping dict("qid" -> AP)
# ap_file_path: Path to the AP file
def store_ap(aps: dict[str, float], ap_output_path: str, append=False):
    os.makedirs(os.path.dirname(ap_output_path), exist_ok=True)
    with open(ap_output_path, "a" if append else "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, ap in aps.items():
            writer.writerow(["map", qid, ap])


# Input:
# weight_file_path: Path to the weight file
# Output: Dictionary containing the mapping dict("qid" -> QueryVector)
# Each QueryVector object is (basically) a dictionary mapping dict("term" -> weight)
def parse_queries(weight_file_path: str) -> dict[str, QueryVector]:
    with open(weight_file_path, "r") as f:
        query_vectors = defaultdict(QueryVector)
        for line in f:
            row = line.strip().split()
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
# Output: Dictionary containing the mapping dict("qid" -> AP)
def parse_ap(ap_file_path: str) -> dict[str, float]:
    aps = dict()
    with open(ap_file_path, "r") as f:
        for line in f:
            row = line.strip().split()
            qid = row[1]
            ap = float(row[2])
            aps[qid] = ap
    return aps


# Input:
# ap_file_path: Path to AP file to append the MAP
def store_mean_ap(ap_file_path: str) -> Union[None, float]:
    aps = parse_ap(ap_file_path)
    if len(aps) == 0:
        return None
    if "all" in aps.keys():
        map = aps["all"]
        print(f"Mean AP: {map}")
        return map
    map = 0.0
    for qid, ap in aps.items():
        map += ap
    map /= len(aps)
    print(f"Mean AP: {map}")
    with open(ap_file_path, "a") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["map", "all", map])
    return map
