import numpy as np
from numpy.linalg import norm

from classes import QueryVector


class Similarities:
    def __init__(self, qid: str, ideal_query: QueryVector, expanded_query: QueryVector):
        self.qid = qid
        self.ideal_query = ideal_query
        self.expanded_query = expanded_query
        self.ideal_query.sort_by_terms()
        self.expanded_query.sort_by_terms()
        self.ideal_query_terms = self.ideal_query.vector.keys()
        self.expanded_query_terms = self.expanded_query.vector.keys()
        self.intersection = self.ideal_query_terms & self.expanded_query_terms
        self.union = self.ideal_query_terms | self.expanded_query_terms
        self.dot_product = sum(
            self.ideal_query[term].weight * self.expanded_query[term].weight
            for term in self.intersection
        )

    def jaccard(self) -> float:
        return len(self.intersection) / len(self.union)

    def jaccard_modified_1(self):
        pass

    def jaccard_modified_2(self):
        pass

    def l1_similarity(self):
        l1_expq = sum(
            [abs(stat.weight) for stat in self.expanded_query.vector.values()]
        )
        return self.dot_product / l1_expq

    def l2_similarity(self):
        l2_expq = norm(
            np.array([stat.weight for stat in self.expanded_query.vector.values()])
        )
        return self.dot_product / l2_expq

    def ndcg(self):
        pass

    def ndcg_modified_1(self):
        pass

    def ndcg_modified_2(self):
        sim = 0.0
        i, j = 0, 0
        for term1, term2 in zip(
            self.ideal_query.vector.keys(), self.expanded_query.vector.keys()
        ):
            if term1 < term2:
                i += 1
            elif term2 > term1:
                j += 1
            else:
                sim += (self.ideal_query.vector[term1].weight * 1000) / (1000 + (j + 1))
                i, j = i + 1, j + 1

        ideal_sim = 0.0
        for i, stat in enumerate(self.ideal_query.vector.values()):
            ideal_sim += (stat.weight * 1000) / (1000 + (i + 1))

        return sim / ideal_sim
