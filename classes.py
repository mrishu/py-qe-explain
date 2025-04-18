import os
import csv
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from typing import Union
import pickle

from definitions import CONTENTS_FIELD


## Lucene imports
from org.apache.lucene.index import Term
from org.apache.lucene.search import TermQuery
from org.apache.lucene.search import BoostQuery
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import BooleanClause


class TRECQuery:
    def __init__(self, qid, text, narr=None) -> None:
        self.qid = qid
        self.text = text
        self.narr = narr

    def __str__(self) -> str:
        return f"{self.qid}\t{self.text}\t{self.narr}"


class QueryVector:
    def __init__(self, vector: Union[dict[str, SimpleNamespace], None] = None) -> None:
        if not vector:
            vector = defaultdict(SimpleNamespace)
        self.vector = vector
        for stat in self.vector.values():
            assert hasattr(stat, "weight")

    def __setitem__(self, term: str, stat: SimpleNamespace) -> None:
        assert hasattr(stat, "weight")
        self.vector[term] = stat

    def __getitem__(self, term: str) -> SimpleNamespace:
        return self.vector[term]

    def __contains__(self, term) -> bool:
        return term in self.vector

    def __len__(self):
        return len(self.vector)

    def store(self, qid, store_path: str, append=True, store_positive=True) -> None:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.sort()  # always sort according to weight before storing
        with open(store_path, "a" if append else "w") as store_file:
            writer = csv.writer(store_file, delimiter="\t")
            for term, stat in self.vector.items():
                if store_positive:
                    if stat.weight > 0:
                        writer.writerow([qid, term, stat.weight])
                else:
                    writer.writerow([qid, term, stat.weight])

    def store_raw(self, store_path: str):
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        with open(store_path, "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    def to_boolquery(self) -> BooleanQuery:
        bool_query_builder = BooleanQuery.Builder()
        for term, rocchio_stat in self.vector.items():
            weight = rocchio_stat.weight
            term_query = TermQuery(Term(CONTENTS_FIELD, term))
            term_query = BoostQuery(term_query, float(weight))
            bool_query_builder.add(term_query, BooleanClause.Occur.SHOULD)
        return bool_query_builder.build()

    def sort(self, sortkey=lambda stat: stat.weight) -> None:
        self.vector = OrderedDict(
            sorted(self.vector.items(), key=lambda x: sortkey(x[1]), reverse=True)
        )

    def sort_by_terms(self):  # alphabetical sorting
        self.vector = OrderedDict(sorted(self.vector.items(), key=lambda x: x[0]))

    def trim(self, num_keep_terms: int = 200) -> None:
        self.vector = dict(list(self.vector.items())[:num_keep_terms])

    def __repr__(self):
        final_str = "Query Vector:\n"
        x = 0
        for term, stat in self.vector.items():
            final_str += f"{term:20s}{stat.weight:.3f}\n"
            x += 1
            if x >= 200:
                break
        return final_str

    def __add__(self, other: "QueryVector") -> "QueryVector":
        added_vector = defaultdict(SimpleNamespace)
        for term, stat in self.vector.items():
            added_vector[term].weight = stat.weight
        for term, stat in other.vector.items():
            if term in added_vector:
                added_vector[term].weight += stat.weight
            else:
                added_vector[term].weight = stat.weight
        return QueryVector(added_vector)

    def __matmul__(self, other: "QueryVector") -> float:
        dot_product = 0
        intersection_terms = self.vector.keys() & other.vector.keys()
        for term in intersection_terms:
            dot_product += self[term].weight * other[term].weight
        return dot_product

    def __truediv__(self, other: Union[float, int]) -> "QueryVector":
        if other == 0.0:
            raise ZeroDivisionError
        div_vector = defaultdict(SimpleNamespace)
        for term in self.vector.keys():
            div_vector[term].weight = self.vector[term].weight / other
        return QueryVector(div_vector)
