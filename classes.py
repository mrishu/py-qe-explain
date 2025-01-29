import os
import csv
from collections import OrderedDict
from types import SimpleNamespace
from typing import Union

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
            vector = dict()
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

    def store(self, qid, store_path: str, append=True) -> None:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.sort()  # always sort according to weight before storing
        with open(store_path, "a" if append else "w") as store_file:
            writer = csv.writer(store_file, delimiter="\t")
            for term, stat in self.vector.items():
                writer.writerow([qid, term, stat.weight])

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

    def display(self, max_terms=200) -> None:
        print("Query Vector:")
        x = 0
        for term, stat in self.vector.items():
            print(f"{term:20s}{stat.weight:.3f}")
            x += 1
            if x >= max_terms:
                break
