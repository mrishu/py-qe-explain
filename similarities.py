from types import SimpleNamespace
import numpy as np
from numpy.linalg import norm
from utils import parse_queries
import csv
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from classes import QueryVector


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


"""How to use?
1. Load the Similarities object with the ideal query as:
    sim_obj = Similarities(qid, ideal_query).
2. Load the expanded query with sim_obj.change_expanded_query(expanded_query).
3. Get similarity using sim_obj.compute_similarity(similarity_name).
4. 
"""


class Similarities:
    def __init__(self, qid, ideal_query: QueryVector):
        self.qid = qid
        self.ideal_query = ideal_query

    def change_expanded_query(self, expanded_query: QueryVector):
        self.expanded_query = expanded_query
        ideal_query_terms = self.ideal_query.vector.keys()
        expanded_query_terms = self.expanded_query.vector.keys()
        self.intersection = ideal_query_terms & expanded_query_terms
        self.union = ideal_query_terms | expanded_query_terms
        self.dot_product = sum(
            self.ideal_query[term].weight * self.expanded_query[term].weight
            for term in self.intersection
        )
        self.l2_idealq = norm(
            np.array([stat.weight for stat in self.ideal_query.vector.values()])
        )
        self.l1_idealq = sum(
            [abs(stat.weight) for stat in self.ideal_query.vector.values()]
        )

    def print_matches(self, outfile_path: str, append=True):
        assert hasattr(self, "expanded_query")
        ideal_query_term_rank = dict()
        expanded_query_term_rank = dict()
        for i, term in enumerate(self.ideal_query.keys()):
            ideal_query_term_rank[term] = i + 1
        for i, term in enumerate(self.expanded_query.keys()):
            expanded_query_term_rank[term] = i + 1

        with open(outfile_path, "a" if append else "w") as file:
            writer = csv.writer(file, delimiter="\t")
            for term in sorted(
                self.intersection, key=lambda term: expanded_query_term_rank[term]
            ):
                writer.writerow(
                    [
                        self.qid,
                        term,
                        self.expanded_query[term].weight,
                        expanded_query_term_rank[term],
                        self.ideal_query[term].weight,
                        ideal_query_term_rank[term],
                    ]
                )
            writer.writerow([])
            writer.writerow([])

    def jaccard(self) -> float:
        assert hasattr(self, "expanded_query")
        return len(self.intersection) / len(self.union)

    def jaccard_modified_1(self):
        assert hasattr(self, "expanded_query")
        pass

    def jaccard_modified_2(self):
        assert hasattr(self, "expanded_query")
        pass

    def l1_similarity(self):
        assert hasattr(self, "expanded_query")
        l1_expq = sum(
            [abs(stat.weight) for stat in self.expanded_query.vector.values()]
        )
        return self.dot_product / (self.l1_idealq * l1_expq)

    def l2_similarity(self):
        assert hasattr(self, "expanded_query")
        l2_expq = norm(
            np.array([stat.weight for stat in self.expanded_query.vector.values()])
        )
        return self.dot_product / (self.l2_idealq * l2_expq)

    def ndcg(self):
        assert hasattr(self, "expanded_query")
        pass

    def ndcg_modified_1(self):
        assert hasattr(self, "expanded_query")
        pass

    def ndcg_modified_2(self):
        assert hasattr(self, "expanded_query")
        self.ideal_query.sort_by_stat()
        self.expanded_query.sort_by_stat()

        ideal_query_term_rank = dict()
        for i, term in enumerate(self.ideal_query.keys()):
            ideal_query_term_rank[term] = i

        sim = 0.0
        for term in self.intersection:
            i = ideal_query_term_rank[term]
            sim += (self.ideal_query[term].weight * 1000) / (1000 + (i + 1))

        ideal_sim = 0.0
        for i, stat in enumerate(self.ideal_query.vector.values()):
            ideal_sim += (stat.weight * 1000) / (1000 + (i + 1))

        return sim / ideal_sim

    def intersection_sim(self) -> float:
        assert hasattr(self, "expanded_query")
        return len(self.intersection)

    def js_similarity(self, log_base=2) -> float:
        vec1 = np.array(
            [
                self.ideal_query.vector.get(term, SimpleNamespace(weight=0.0)).weight
                for term in self.intersection
            ],
            dtype=np.float64,
        )
        vec2 = np.array(
            [
                self.expanded_query.vector.get(term, SimpleNamespace(weight=0.0)).weight
                for term in self.intersection
            ],
            dtype=np.float64,
        )

        # Clip small values to avoid numerical issues
        epsilon = 1e-12
        vec1 = np.clip(vec1, epsilon, None)
        vec2 = np.clip(vec2, epsilon, None)

        # Compute Jensen-Shannon divergence
        js_dist = jensenshannon(vec1, vec2, base=log_base)
        return 1 - js_dist**2

    def wasserstein_similarity(self, lamda=1) -> float:
        assert hasattr(self, "expanded_query")
        self.intersection = sorted(
            self.intersection, key=lambda x: self.ideal_query[x].weight, reverse=True
        )
        if len(self.intersection) == 0:
            return 0.0
        vec1 = np.array(
            [
                self.ideal_query.vector.get(term, SimpleNamespace(weight=0.0)).weight
                for term in self.intersection
            ],
            dtype=np.float64,
        )
        vec2 = np.array(
            [
                self.expanded_query.vector.get(term, SimpleNamespace(weight=0.0)).weight
                for term in self.intersection
            ],
            dtype=np.float64,
        )

        # Compute Jensen-Shannon divergence
        positions = [x for x in range(1, len(self.intersection) + 1, 1)]
        wd = wasserstein_distance(positions, positions, vec1, vec2)
        return np.exp(-lamda * wd)

    def compute_similarity(self, sim_name: str) -> float:
        assert hasattr(self, "expanded_query")
        sims = {
            "j": "jaccard",
            "l1": "l1_similarity",
            "l2": "l2_similarity",
            "n2": "ndcg_modified_2",
            "inter": "intersection_sim",
            "js": "js_similarity",
            "w": "wasserstein_similarity",
        }
        return getattr(self, sims[sim_name])()


# This __main__ part can be used to explicitly check the similarities of two queries
if __name__ == "__main__":
    ideal_query_weight_file = (
        "./ideal-queries/trec678/weights/ideal_query_chi2_lr_max.term_weights"
    )
    # expanded_query_weight_file = (
    #     "./ideal-queries/trec678/weights/ideal_query.term_weights"
    # )
    expanded_query_weight_file = (
        "./expanded-queries-max-terms/trec678/spl/weights/spl-10-8.term_weights"
    )

    ideal_queries = parse_queries(ideal_query_weight_file)
    expanded_queries = parse_queries(expanded_query_weight_file)

    j = []
    l1 = []
    l2 = []
    n2 = []
    js = []
    inter = []
    w = []
    for qid, ideal_query in ideal_queries.items():
        if qid not in expanded_queries:
            continue
        expanded_query = expanded_queries[qid]
        expanded_query.trim(200)
        ideal_query.trim(200)
        sim = Similarities(qid, ideal_query)
        sim.change_expanded_query(expanded_query)
        j.append(sim.jaccard())
        l1.append(sim.l1_similarity())
        l2.append(sim.l2_similarity())
        n2.append(sim.ndcg_modified_2())
        js.append(sim.js_similarity())
        inter.append(sim.intersection_sim())
        w.append(sim.wasserstein_similarity())
        print(
            qid,
            sim.jaccard(),
            sim.l1_similarity(),
            sim.l2_similarity(),
            sim.ndcg_modified_2(),
            sim.js_similarity(),
            sim.wasserstein_similarity(),
            sim.intersection_sim(),
            len(sim.ideal_query),
            len(sim.expanded_query),
            sep="\t",
        )
    print(
        "avg",
        sum(j) / len(j),
        sum(l1) / len(l1),
        sum(l2) / len(l2),
        sum(n2) / len(n2),
        sum(js) / len(js),
        sum(w) / len(w),
        sum(inter) / len(inter),
        sep="\t",
    )
