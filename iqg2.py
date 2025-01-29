import os
from collections import defaultdict, OrderedDict
import math
import re
import csv
from typing import Union
import xml.etree.ElementTree as ET

import lucene
from java.io import File
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.core import StopFilter
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexReader
from org.apache.lucene.index import LeafReader
from org.apache.lucene.index import LeafReaderContext
from org.apache.lucene.index import PostingsEnum
from org.apache.lucene.index import Term
from org.apache.lucene.index import Terms
from org.apache.lucene.index import TermsEnum
from org.apache.lucene.queryparser.classic import ParseException
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import DocIdSetIterator
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import Query
from org.apache.lucene.search import ScoreDoc
from org.apache.lucene.search import TermQuery
from org.apache.lucene.search import BoostQuery
from org.apache.lucene.search import TopDocs
from org.apache.lucene.search import TopScoreDocCollector
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.search.similarities import BasicStats
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.util import BytesRef
from org.apache.lucene.util import BytesRefIterator

import pytrec_eval
from tqdm import tqdm

lucene.initVM()


class TermStat:
    def __init__(self, docid, lucene_docid, tf, weight):
        self.docid = docid
        self.lucene_docid = lucene_docid
        self.tf = tf
        self.weight = weight

    def __str__(self):
        return f"{self.docid}\t{self.lucene_docid}\t{self.tf}\t{self.weight}"


class RocchioStat:
    def __init__(self, weight, doc_freq_in_rel_docs):
        self.weight = weight
        self.doc_freq_in_rel_docs = doc_freq_in_rel_docs

    def __str__(self):
        return f"{self.weight}\t{self.doc_freq_in_rel_docs}"


class TRECQuery:
    def __init__(self, qid, text, narr) -> None:
        self.qid = qid
        self.text = text
        self.narr = narr

    def __str__(self) -> str:
        return f"{self.qid}\t{self.text}\t{self.narr}"


class IdealQueryGeneration:
    ID_FIELD = "id"
    CONTENT_FIELD = "contents"
    lucene_docid_cache = dict()

    def __init__(self, index_dir: str, qrel_path: str, stopwords_path: str) -> None:
        self.reader = DirectoryReader.open(FSDirectory.open(File(index_dir).toPath()))
        self.storedfields = self.reader.storedFields()
        self.termvectors = self.reader.termVectors()
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(BM25Similarity())
        self.searcher.setMaxClauseCount(4096)
        with open(stopwords_path, "r") as stopwords_file:
            self.stopwords = stopwords_file.read().strip().split()
        print("Stopwords size:", len(self.stopwords))
        self.analyzer = EnglishAnalyzer(StopFilter.makeStopSet(self.stopwords))
        self.query_rel_docs_map = defaultdict(set)
        self.query_non_rel_docs_map = defaultdict(set)
        self.no_of_docs = self.reader.numDocs()
        self.avg_dl = (
            self.reader.getSumTotalTermFreq(self.CONTENT_FIELD) / self.no_of_docs
        )
        with open(qrel_path, "r") as qrel_file:
            self.qrel = pytrec_eval.parse_qrel(qrel_file)
        self.qrel_evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {"map", "ndcg"})

    def raw_search(self, query_text: str, k: int) -> dict[str, float]:
        query_text = re.sub(r'[/|,|"]', " ", query_text)
        query = QueryParser(self.CONTENT_FIELD, self.analyzer).parse(query_text)
        top_docs = self.searcher.search(query, k)
        results = dict()
        for hit in top_docs.scoreDocs:
            doc = self.storedfields.document(hit.doc)
            docid = doc.get(self.ID_FIELD)
            score = hit.score
            results[docid] = score
        return results

    def search(self, query: Query, k: int) -> dict[str, float]:
        top_docs = self.searcher.search(query, k)
        results = dict()
        for hit in top_docs.scoreDocs:
            lucene_docid = hit.doc
            doc = self.storedfields.document(lucene_docid)
            docid = doc.get(self.ID_FIELD)
            score = hit.score
            results[docid] = score
        return results

    def read_qrel(self) -> None:
        for qid, doc_wise_rel in self.qrel.items():
            for docid, rel in doc_wise_rel.items():
                if rel != 0:
                    lucene_docid = self.get_lucene_docid(docid)
                    if lucene_docid != -1:
                        self.query_rel_docs_map[qid].add((docid, lucene_docid))
                else:
                    lucene_docid = self.get_lucene_docid(docid)
                    if lucene_docid != -1:
                        self.query_non_rel_docs_map[qid].add((docid, lucene_docid))

    def get_lucene_docid(self, docid: str) -> int:
        if docid in self.lucene_docid_cache:
            return self.lucene_docid_cache[docid]
        query = TermQuery(Term(self.ID_FIELD, docid))
        hits = self.searcher.search(query, 1).scoreDocs
        if len(hits) == 0:
            # print(f"{docid}: document not found")
            return -1
        lucene_docid = hits[0].doc
        self.lucene_docid_cache[docid] = lucene_docid
        return lucene_docid

    def compute_bm25_weight(
        self, term: str, tf: float, doc_len: float, query_term: bool = False
    ) -> float:
        k1 = 1.2
        b = 0.75
        doc_freq = self.reader.docFreq(Term(self.CONTENT_FIELD, term))
        idf = math.log(1 + (self.no_of_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        if query_term:
            return idf
        return idf * tf / (tf + k1 * (1 - b + b * doc_len / self.avg_dl))

    def get_termstats(
        self, query: TRECQuery, from_relevant_docs: bool
    ) -> dict[str, list[TermStat]]:
        """Given a query, it collects the terms from relevant/non-relevant documents.
        It then returns a dictionary: term (str) -> list[TermStat],
        where each TermStat contains [docid, lucene_docid, tf, and bm25_weight] of the term in document identified by docid.
        """
        termstats = defaultdict(list)  # Maps from "term" -> List[TermStat]
        if from_relevant_docs:
            if query.qid in self.query_rel_docs_map:
                docids = self.query_rel_docs_map[query.qid]
            else:
                return termstats
        else:
            if query.qid in self.query_non_rel_docs_map:
                docids = self.query_non_rel_docs_map[query.qid]
            else:
                return termstats
        tqdm.write(f"No. of docids to traverse: {len(docids)}")
        for docid, lucene_docid in docids:
            tvec = self.termvectors.get(lucene_docid, self.CONTENT_FIELD)
            docLen = tvec.getSumTotalTermFreq()
            tvec_iter = tvec.iterator()
            for term in BytesRefIterator.cast_(tvec_iter):
                term_str = term.utf8ToString()
                tf = tvec_iter.totalTermFreq()
                weight = self.compute_bm25_weight(term_str, tf, docLen, False)
                termstats[term_str].append(TermStat(docid, lucene_docid, tf, weight))
        tqdm.write(f"No. of terms collected: {len(termstats)}")
        return termstats

    def compute_rocchio_vector(
        self,
        query: TRECQuery,
        beta: float,
        gamma: float,
    ) -> dict[str, RocchioStat]:
        query_termstats_rel = self.get_termstats(query, True)
        query_termstats_non_rel = self.get_termstats(query, False)
        rocchio_vector = dict()
        for term, rel_stats in query_termstats_rel.items():
            rel_avg_weight = sum(stat.weight for stat in rel_stats) / max(
                1, len(query_termstats_rel)
            )
            non_rel_stats = query_termstats_non_rel[term]
            non_rel_avg_weight = sum(stat.weight for stat in non_rel_stats) / max(
                1, len(query_termstats_non_rel)
            )
            weight = beta * rel_avg_weight - gamma * non_rel_avg_weight
            rocchio_vector[term] = RocchioStat(weight, len(query_termstats_rel[term]))
        return rocchio_vector

    def store_expanded_query(
        self,
        query: TRECQuery,
        rocchio_vector: dict[str, RocchioStat],
        alpha: float,
        store_path: str,
    ) -> None:
        query_terms = set(self._get_query_terms(query))
        # Add alpha * bm25_weight of the query to the rocchio vector
        for term in query_terms:
            weight = alpha * self.compute_bm25_weight(term, 1, len(query_terms), True)
            if term not in rocchio_vector:
                rocchio_vector[term] = RocchioStat(
                    weight, 0
                )  # the second entry (doc_freq_in_rel_docs) is irrelevant here,
                # as we need to sort by weight before storing
            else:
                rocchio_vector[term].weight += weight
        rocchio_vector = self.sort_rocchio_vector(rocchio_vector)
        with open(store_path, "a") as store_file:
            writer = csv.writer(store_file, delimiter="\t")
            for term, rocchio_stat in rocchio_vector.items():
                if not term:
                    continue
                weight = rocchio_stat.weight
                if weight < 0:  # skip if weight < 0
                    continue
                writer.writerow([query.qid, term, weight])

    def _get_query_terms(self, query: TRECQuery) -> list[str]:
        query_text = re.sub(r'[/|,|"]', " ", query.text)
        parsed_query = QueryParser(self.CONTENT_FIELD, self.analyzer).parse(query_text)
        query_terms = parsed_query.toString(self.CONTENT_FIELD).strip().split()
        return query_terms

    def expanded_query(
        self, query: TRECQuery, rocchio_vector: dict, alpha: float
    ) -> BooleanQuery:
        query_terms = set(self._get_query_terms(query))
        bool_query_builder = BooleanQuery.Builder()
        for term in query_terms:
            weight = alpha * self.compute_bm25_weight(term, 1, len(query_terms), True)
            term_query = TermQuery(Term(self.CONTENT_FIELD, term))
            term_query = BoostQuery(term_query, float(weight))
            bool_query_builder.add(term_query, BooleanClause.Occur.SHOULD)
        for term, rocchio_stat in rocchio_vector.items():
            if not term:
                continue
            weight = rocchio_stat.weight
            if weight < 0:  # skip if weight < 0
                continue
            term_query = TermQuery(Term(self.CONTENT_FIELD, term))
            term_query = BoostQuery(term_query, float(weight))
            bool_query_builder.add(term_query, BooleanClause.Occur.SHOULD)
        return bool_query_builder.build()

    def computeMAP(
        self,
        query: TRECQuery,
        rocchio_vector: dict[str, RocchioStat],
        alpha: float,
        store_run_path: Union[str, None] = None,
        runid="idealQuery",
    ) -> float:
        boolean_expanded_query = self.expanded_query(query, rocchio_vector, alpha)
        run = dict()
        run[query.qid] = self.search(boolean_expanded_query, 1000)
        if store_run_path:
            with open(store_run_path, "a") as run_file:
                writer = csv.writer(run_file, delimiter="\t")
                rank = 0
                for docid, score in run[query.qid].items():
                    writer.writerow([query.qid, "Q0", docid, rank, score, runid])
                    rank += 1
        return self.qrel_evaluator.evaluate(run)[query.qid]["map"]

    def trim_rocchio_vector(
        self,
        rocchio_vector: OrderedDict[str, RocchioStat],
        num_keep_terms: int = 200,
    ) -> OrderedDict[str, RocchioStat]:
        return OrderedDict(list(rocchio_vector.items())[:num_keep_terms])

    def sort_rocchio_vector(
        self,
        rocchio_vector: dict[str, RocchioStat],
        sortkey=lambda rocchio_stat: rocchio_stat.weight,
    ) -> OrderedDict[str, RocchioStat]:
        return OrderedDict(
            sorted(rocchio_vector.items(), key=lambda x: sortkey(x[1]), reverse=True)
        )

    def tweak_rocchio_weight_vector(
        self,
        query: TRECQuery,
        rocchio_vector: dict[str, RocchioStat],
        alpha: float,
        tweak_magnitude_list: list[float] = [4.0, 2.0, 1.0, 0.5, 0.25],
    ) -> dict[str, RocchioStat]:
        for mag in tweak_magnitude_list:
            for term, stat in tqdm(
                rocchio_vector.items(), desc=f"Tweak Magnitude: {mag}"
            ):
                current_weight = stat.weight
                current_map = self.computeMAP(query, rocchio_vector, alpha)
                tqdm.write(
                    f"Term: {term:20s}Current Weight: {current_weight:.3f}, Current MAP: {current_map:.3f}"
                )
                nudged_weight = (1 + mag) * current_weight
                rocchio_vector[term].weight = nudged_weight
                nudged_map = self.computeMAP(query, rocchio_vector, alpha)
                if nudged_map >= current_map:
                    tqdm.write(
                        f"Term: {term:20s}Nudged  Weight: {nudged_weight:.3f}, Nudged  MAP: {nudged_map:.3f} -- Keeping   Nudged Weight"
                    )
                else:
                    tqdm.write(
                        f"Term: {term:20s}Nudged  Weight: {nudged_weight:.3f}, Nudged  MAP: {nudged_map:.3f} -- Reverting Nudged Weight"
                    )
                    rocchio_vector[term].weight = current_weight
        return rocchio_vector

    def close(self):
        self.reader.close()


def display_rocchio_vector(rocchio_vector: dict[str, RocchioStat], max_terms=200):
    print("Rocchio Vector:")
    x = 0
    for term, stat in rocchio_vector.items():
        print(f"{term:20s}{stat.weight:.3f}")
        x += 1
        if x >= max_terms:
            break


# Example usage
if __name__ == "__main__":
    index_path = "index-dir/trec678rb"
    queries_file = "trec678rb/topics/trec678rb.xml"
    qrel_path = "trec678rb/qrels/trec678rb.qrel"
    stopwords_path = "resources/smart-stopwords"

    run_file = "idealQuery2.run"
    weights_store_file = "idealQuery2.weights"

    alpha = 2.0
    beta = 64.0
    gamma = 64.0

    NUM_QUERIES_TO_PROCESS = 20
    NUM_EXPANSION_TERMS = 200

    iqg = IdealQueryGeneration(index_path, qrel_path, stopwords_path)
    print("Index initialized.")

    ## Read qrel file's info into dictionaries for future use
    print("Reading Qrel File...")
    iqg.read_qrel()

    # Example
    # query = TRECQuery(
    #     "301", "international organized crime", "some irrelevant explanation"
    # )

    i = 0
    robust_topics = ET.parse(queries_file).getroot()
    for top in tqdm(robust_topics, desc="Queries Processed"):
        qid = top[0].text.strip()  # this is query number
        query_text = top[1].text.strip()  # this will be our query
        narr = top[2].text.strip()

        query = TRECQuery(qid, query_text, narr)
        tqdm.write(str(query))

        ## STEP 1: Compute initial rocchio vector
        tqdm.write("Computing Rocchio Vector...")
        query_rocchio_vector = iqg.compute_rocchio_vector(query, beta, gamma)

        ## STEP 2: Remove rare terms from
        tqdm.write("Removing rare terms...")
        # Sort it according to decreasing order of frequency in relevant documents
        query_rocchio_vector = iqg.sort_rocchio_vector(
            query_rocchio_vector, lambda stat: stat.doc_freq_in_rel_docs
        )
        num_valid_terms = 0
        min_threshold = 0.02 * len(iqg.query_rel_docs_map[query.qid])
        for term, stat in query_rocchio_vector.items():
            if stat.doc_freq_in_rel_docs < min_threshold:
                break
            num_valid_terms += 1
        query_rocchio_vector = iqg.trim_rocchio_vector(
            query_rocchio_vector, num_valid_terms
        )

        ## STEP 3: Sort it according to decreasing order of weights
        query_rocchio_vector = iqg.sort_rocchio_vector(query_rocchio_vector)

        ## STEP 4: Trimming rocchio to terms with top 200 weights
        tqdm.write(f"Trimming Rocchio Vector to top {NUM_EXPANSION_TERMS} terms.")
        query_rocchio_vector = iqg.trim_rocchio_vector(
            query_rocchio_vector, NUM_EXPANSION_TERMS
        )

        ## STEP 5: Trimming rocchio vector to reomve negative weights
        tqdm.write("Trimming Rocchio Vector to keep positive weight terms.")
        num_pos_terms = 0
        for term, stat in query_rocchio_vector.items():
            if stat.weight < 0:
                break
            num_pos_terms += 1
        query_rocchio_vector = iqg.trim_rocchio_vector(
            query_rocchio_vector, num_pos_terms
        )

        ## STEP 6: Start tweaking
        query_rocchio_vector = iqg.tweak_rocchio_weight_vector(
            query, query_rocchio_vector, alpha
        )

        ## STEP 7: Store final run
        tqdm.write(
            f"Final MAP: {iqg.computeMAP(query, query_rocchio_vector, alpha, store_run_path=run_file):.3f}"
        )

        ## STEP 8: Store expanded query
        iqg.store_expanded_query(query, query_rocchio_vector, alpha, weights_store_file)

        i += 1
        if i >= NUM_QUERIES_TO_PROCESS:
            break

    iqg.close()
