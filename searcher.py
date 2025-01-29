## Library imports
import re
import csv
from collections import OrderedDict

## Extra imports
import pytrec_eval

## Project imports
from definitions import (
    ID_FIELD,
    CONTENTS_FIELD,
    ROOT_DIR,
    TREC_INDEX_DIR_PATH,
    STOPWORDS_FILE_PATH,
    TREC_QREL_FILE_PATH,
)
from classes import QueryVector
from utils import store_run

## Lucene imports
from java.io import File
from org.apache.lucene.analysis.core import StopFilter
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import Query
from org.apache.lucene.search import TermQuery
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.store import FSDirectory


class SearchAndEval:
    lucene_docid_cache = dict()

    def __init__(self, index_dir: str, stopwords_path: str, qrel_path: str) -> None:
        self.reader = DirectoryReader.open(FSDirectory.open(File(index_dir).toPath()))
        self.storedfields = self.reader.storedFields()
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(BM25Similarity())
        self.searcher.setMaxClauseCount(
            4096
        )  # since the expanded queries can get very long
        with open(stopwords_path, "r") as stopwords_file:
            stopwords = stopwords_file.read().strip().split()
        self.analyzer = EnglishAnalyzer(StopFilter.makeStopSet(stopwords))
        with open(qrel_path, "r") as qrel_file:
            self.actual_qrel = pytrec_eval.parse_qrel(qrel_file)
        self.qrel_evaluator = pytrec_eval.RelevanceEvaluator(self.actual_qrel, {"map"})

    def raw_search(self, query_text: str, k: int) -> dict[str, float]:
        query_text = re.sub(r'[/|,|"]', " ", query_text)
        query = QueryParser(CONTENTS_FIELD, self.analyzer).parse(query_text)
        top_docs = self.searcher.search(query, k)
        results = OrderedDict()  # results are already sorted by score
        for hit in top_docs.scoreDocs:
            doc = self.storedfields.document(hit.doc)
            docid = doc.get(ID_FIELD)
            score = hit.score
            results[docid] = score
        return results

    def search(self, query: Query, k: int = 1000) -> dict[str, float]:
        top_docs = self.searcher.search(query, k)
        results = OrderedDict()  # results are already sorted by score
        for hit in top_docs.scoreDocs:
            lucene_docid = hit.doc
            doc = self.storedfields.document(lucene_docid)
            docid = doc.get(ID_FIELD)
            score = hit.score
            results[docid] = score
        return results

    def computeAP(self, qid: str, query_vector: QueryVector, num_top_docs=1000):
        run = dict()
        results = self.search(query_vector.to_boolquery(), num_top_docs)
        run[qid] = results
        eval = self.qrel_evaluator.evaluate(run)
        if qid not in eval:
            return None, run
        else:
            return eval[qid]["map"], run

    def get_lucene_docid(self, docid: str) -> int:
        if docid in self.lucene_docid_cache:
            return self.lucene_docid_cache[docid]
        query = TermQuery(Term(ID_FIELD, docid))
        hits = self.searcher.search(query, 1).scoreDocs
        if len(hits) == 0:
            return -1
        lucene_docid = hits[0].doc
        self.lucene_docid_cache[docid] = lucene_docid
        return lucene_docid

    def close(self):
        self.reader.close()


if __name__ == "__main__":
    searcher = SearchAndEval(
        TREC_INDEX_DIR_PATH, STOPWORDS_FILE_PATH, TREC_QREL_FILE_PATH
    )
    extracted_queries_path = f"{ROOT_DIR}/extracted-queries/trec678"
    run_output_path = f"{ROOT_DIR}/test-runs/bm25.run"

    reader = csv.reader(open(extracted_queries_path, "r"), delimiter="\t")
    run = dict()
    for row in reader:
        qid = row[0]
        query_text = row[1]
        run[qid] = searcher.raw_search(query_text, 1000)
        print(f"Stored run for query {qid}")
    store_run(run, run_output_path, runid="bm25", append=True)
