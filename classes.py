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

    def __repr__(self) -> str:
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

    def __len__(self) -> int:
        return len(self.vector)

    def __iter__(self):
        return self.vector.__iter__()

    def items(self):
        return self.vector.items()

    def values(self):
        return self.vector.values()

    def keys(self):
        return self.vector.keys()

    def store_txt(self, qid, store_path: str, append=True) -> None:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.sort_by_stat()  # always sort according to weight before storing
        with open(store_path, "a" if append else "w") as store_file:
            writer = csv.writer(store_file, delimiter="\t")
            for term, stat in self.vector.items():
                writer.writerow([qid, term, stat.weight])

    def store_pickle(self, store_path: str) -> None:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.sort_by_stat()  # always sort according to weight before storing
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

    # default: sort by weight
    def sort_by_stat(self, sortkey=lambda stat: stat.weight, reverse=True) -> None:
        self.vector = OrderedDict(
            sorted(self.vector.items(), key=lambda x: sortkey(x[1]), reverse=reverse)
        )

    def sort_by_terms(self) -> None:  # alphabetical sorting
        self.vector = OrderedDict(sorted(self.vector.items(), key=lambda x: x[0]))

    def trim(self, num_keep_terms: int = 200) -> None:
        self.vector = dict(list(self.vector.items())[:num_keep_terms])

    def __repr__(self) -> str:
        final_str = "Query Vector:\n"
        x = 0
        for term, stat in self.vector.items():
            final_str += f"{term:20s}{stat}\n"
            x += 1
            if x >= 200:  # display only upto 200 terms
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

    def __sub__(self, other: "QueryVector") -> "QueryVector":
        added_vector = defaultdict(SimpleNamespace)
        for term, stat in self.vector.items():
            added_vector[term].weight = stat.weight
        for term, stat in other.vector.items():
            if term in added_vector:
                added_vector[term].weight += stat.weight
            else:
                added_vector[term].weight = stat.weight
        return QueryVector(added_vector)

    def __mul__(self, other: Union[float, int]) -> "QueryVector":
        mul_vector = defaultdict(SimpleNamespace)
        for term in self.vector.keys():
            mul_vector[term].weight = self.vector[term].weight * other
        return QueryVector(mul_vector)

    def __truediv__(self, other: Union[float, int]) -> "QueryVector":
        if other == 0.0:
            raise ZeroDivisionError
        div_vector = defaultdict(SimpleNamespace)
        for term in self.vector.keys():
            div_vector[term].weight = self.vector[term].weight / other
        return QueryVector(div_vector)

    def remove_non_positive_weights(self) -> None:
        new_vector = OrderedDict()
        for term, stat in self.vector.items():
            if stat.weight <= 0:
                continue
            new_vector[term] = stat
        self.vector = new_vector
