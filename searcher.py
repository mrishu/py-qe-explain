## Library imports
import re
import csv
from collections import OrderedDict, defaultdict
import os
from copy import deepcopy

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
from utils import store_run, store_ap, store_mean_ap

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
            8192
        )  # since the expanded queries can get very long
        with open(stopwords_path, "r") as stopwords_file:
            stopwords = stopwords_file.read().strip().split()
        self.analyzer = EnglishAnalyzer(StopFilter.makeStopSet(stopwords))
        with open(qrel_path, "r") as qrel_file:
            self.qrel = pytrec_eval.parse_qrel(qrel_file)
        self.qrel_evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {"map"})

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

    def computeAP_and_run(self, qid: str, query_vector: QueryVector, num_top_docs=1000):
        qvec = deepcopy(query_vector)
        qvec.remove_non_positive_weights()
        results = self.search(qvec.to_boolquery(), num_top_docs)
        run = dict()
        run[qid] = results
        ap = self.get_APs_from_run(run)[qid]
        return ap, run

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

    def get_APs_from_run(self, run):
        aps = defaultdict(None)
        eval = self.qrel_evaluator.evaluate(run)
        for qid in run.keys():
            if qid not in eval:
                aps[qid] = None
            else:
                aps[qid] = eval[qid]["map"]
        return aps

    def close(self):
        self.reader.close()


if __name__ == "__main__":
    # the below script generates a run file for a BM25 top 1000 retrieval
    searcher = SearchAndEval(
        TREC_INDEX_DIR_PATH, STOPWORDS_FILE_PATH, TREC_QREL_FILE_PATH
    )
    extracted_queries_path = os.path.join(ROOT_DIR, "extracted-queries", "trec678")
    run_output_path = os.path.join(ROOT_DIR, "test-runs", "bm25.run")
    ap_output_path = os.path.join(ROOT_DIR, "test-runs", "bm25.ap")

    if os.path.exists(run_output_path):
        os.remove(run_output_path)
    if os.path.exists(run_output_path):
        os.remove(ap_output_path)

    reader = csv.reader(open(extracted_queries_path, "r"), delimiter="\t")
    run = dict()
    for row in reader:
        qid = row[0]
        query_text = row[1]
        run[qid] = searcher.raw_search(query_text, 1000)
        print(f"Stored run for query {qid}")
    store_run(run, run_output_path, runid="bm25", append=True)
    aps = searcher.get_APs_from_run(run)
    store_ap(aps, ap_output_path, append=True)
    store_mean_ap(ap_output_path)
