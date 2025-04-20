from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path

ROOT_DIR = os.path.join(
    Path(os.path.dirname(os.path.realpath(__file__))).absolute().parent
)


@dataclass
class Pair:
    runid: int
    similarity: float
    ap: float
    rank_s: int
    rank_a: int


# ---------------------------------------------------------------------------#
# File names containing ideal / expanded query term-weights

ideal_q_runid = "ideal_query_restrict"

term_weight_file = dict()
term_weight_file["ideal"] = (
    ROOT_DIR + f"/ideal-queries/trec678/weights/{ideal_q_runid}.term_weights"
)

# Files containing AP achieved by ideal / expanded query
ap_file = dict()
ap_file["ideal"] = ROOT_DIR + f"/ideal-queries/trec678/aps/{ideal_q_runid}.ap"

# Aliases
ideal_query_path = term_weight_file["ideal"]
ideal_ap_file_path = ap_file["ideal"]

expanded_query_path = ROOT_DIR + "/expanded-queries/trec678/"
# ---------------------------------------------------------------------------#

# COLLECTIONS, EXPANSION METHODS, PARAMETERS

# Which ones do we want to experiment with for now?
# colls = [ "trec678", "clueweb09b" ]
colls = ["trec678"]
expansion_methods = ["rm3", "ceqe", "loglogistic", "spl"]
# expansion_methods = ["spl"]
num_top_docs = [10, 20, 30, 40]  # [10, 20, 30, 40, 50, 60]
num_exp_terms = [15, 25, 35, 45, 55]  # [15, 25, 35, 45, 55, 65, 75]

# mixing_param = 0.5

num_correlation_pairs = len(expansion_methods) * len(num_top_docs) * len(num_exp_terms)

method_params = {
    "rm3": "0.6",
    "ceqe": ".6",
    "loglogistic": "2",
    "spl": "8",
}
# ---------------------------------------------------------------------------#

runids = [
    "-".join([exp_method, str(num_terms), str(num_docs), method_params[exp_method]])
    for exp_method, num_docs, num_terms in product(
        expansion_methods, num_top_docs, num_exp_terms
    )
]
