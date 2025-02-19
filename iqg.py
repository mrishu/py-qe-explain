## Library imports
import os
from collections import defaultdict
import math
import re
from types import SimpleNamespace
from typing import Union
import xml.etree.ElementTree as ET
import csv
import argparse
import pickle

## Extra imports
import pytrec_eval
from tqdm import tqdm

## Local Imports
from definitions import (
    CONTENTS_FIELD,
    ROOT_DIR,
    TREC_INDEX_DIR_PATH,
    TREC_QREL_FILE_PATH,
    STOPWORDS_FILE_PATH,
)
from classes import (
    TRECQuery,
    QueryVector,
)
from utils import store_run
from searcher import SearchAndEval

## Lucene imports
from org.apache.lucene.index import Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.util import BytesRefIterator

"""
# Overall Procedure:
1. Construct `query_rel_docs_map`, `query_non_rel_docs_map` dictionaries
    using the function `_read_restrict_qrel()` function
    from the `restrict_qrel_path` file.
    - `query_rel_docs_map` is a dictionary mapping from `qid` -> `list[docid]` of relevant documents.
    - `query_non_rel_docs_map` is a dictionary mapping from `qid` -> `list[docid]` of non-relevant documents.
    These two dictionaries are computed for the whole `IdealQueryGeneration` instance once.
2. For each query, initial Rocchio vector is computed using the `compute_rocchio_vector()` function.
    - The Rocchio vector will essentially be a dictionary mapping from `term` -> `weight`.
      And this `weight` will be computed as:
      ```
       alpha * term BM25 weight in query
       + beta * average of term BM25 weights in relevant documents
       - gamma * average of term BM25 weights in non-relevant documents,
      ```
      where `term`s ideally varies over all possible terms.
      But since, weight < 0 is not allowed, we can safely ignore terms that don't occur in the query or the relevant documents.
3. Inside `compute_rocchio_vector()` function:
    1. We first invoke the `_get_termstats()` function to get the "term statistics" from relevant/non-relevant documents.
        This function collects all the terms occuring in the relevant/non-relevant documents and then
        returns dictionaries mapping from "term" -> list[(docid, weight)] for relevant and non-relevant documents separately.
        Here `weight` is the BM25 weight of the `term` in the document identified by `docid`.
        These are collected as `query_termstats_rel` and `query_termstats_non_rel` dictionaries for relevant/non-relevant documents respectively.
    2. We then go over all `term`s in `query_termstats_rel` (as we are considering only `term`s that occur in relevant documents)
       and compute the Rocchio vector `weight` of `term` by:
       ```
       rel_avg_weight = avg(weights in query_termstats_rel[term])
       non_rel_avg_weight = avg(weights in query_termstats_non_rel[term])
       rocchio_vector[term] = alpha * weight of term in query + beta * rel_avg_weight - gamma * non_rel_avg_weight
       ```
    3. This dictionary `rocchio_vector` is then returned.
4. After the `rocchio_vector` for a query is computed, we tweak the weights of each `term` to increase the MAP as much as possible.
    We select a `tweak_magnitude` from the `tweak_magnitude_list = [4.0, 2.0, 1.0, 0.5, 0.25]`. Then,
        - We tweak the `weight` of each `term` by multipyling it by `(1 + tweak_magnitude)`.  
            If the MAP increases, we keep the modified `weight` otherwise we restore it.
        - At the end after all `term`s have been processed, we select the next `tweak_magnitude` and repeat the above process.
    After all `tweak_magnitude`s from `tweak_magnitude_list` have been processed, we return the final tweaked Rocchio vector
    as the "Ideal Query" for the query.
"""


class IdealQueryGeneration(SearchAndEval):

    def __init__(
        self,
        index_dir: str,
        stopwords_path: str,
        actual_qrel_path: str,
        restrict_qrel_path: Union[str, None] = None,
    ) -> None:
        super().__init__(index_dir, stopwords_path, actual_qrel_path)
        self.termvectors = self.reader.termVectors()
        self.no_of_docs = self.reader.numDocs()
        self.avg_dl = self.reader.getSumTotalTermFreq(CONTENTS_FIELD) / self.no_of_docs
        self.query_rel_docs_map = defaultdict(set)
        self.query_non_rel_docs_map = defaultdict(set)
        if restrict_qrel_path is None:
            restrict_qrel_path = actual_qrel_path
        with open(restrict_qrel_path, "r") as qrel_file:
            self.restrict_qrel = pytrec_eval.parse_qrel(qrel_file)
        if os.path.exists(restrict_qrel_path + ".pkl"):
            self.query_rel_docs_map, self.query_non_rel_docs_map = pickle.load(
                open(restrict_qrel_path + ".pkl", "rb")
            )
        else:
            self._read_restrict_qrel()
            pickle.dump(
                (self.query_rel_docs_map, self.query_non_rel_docs_map),
                open(restrict_qrel_path + ".pkl", "wb"),
            )

    def _read_restrict_qrel(self) -> None:
        for qid, doc_wise_rel in self.restrict_qrel.items():
            for docid, rel in doc_wise_rel.items():
                if rel != 0:
                    lucene_docid = self.get_lucene_docid(docid)
                    if lucene_docid != -1:
                        self.query_rel_docs_map[qid].add((docid, lucene_docid))
                else:
                    lucene_docid = self.get_lucene_docid(docid)
                    if lucene_docid != -1:
                        self.query_non_rel_docs_map[qid].add((docid, lucene_docid))

    def _get_query_terms(self, query: TRECQuery) -> list[str]:
        query_text = re.sub(r'[/|,|"]', " ", query.text)
        parsed_query = QueryParser(CONTENTS_FIELD, self.analyzer).parse(query_text)
        query_terms = parsed_query.toString(CONTENTS_FIELD).strip().split()
        return query_terms

    def _compute_bm25_weight(
        self, term: str, tf: float, doc_len: float, is_query_term: bool = False
    ) -> float:
        k1 = 1.2
        b = 0.75
        doc_freq = self.reader.docFreq(Term(CONTENTS_FIELD, term))
        idf = math.log(1 + (self.no_of_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        if is_query_term:
            return idf
        return idf * tf / (tf + k1 * (1 - b + b * doc_len / self.avg_dl))

    def _get_termstats(
        self, query: TRECQuery, from_relevant_docs: bool
    ) -> dict[str, list[SimpleNamespace]]:
        """Given a query, it collects the terms from relevant/non-relevant documents.
        It then returns a dictionary: term (str) -> list[SimpleNamespace],
        where each SimpleNamespace has term statistics containing
        [docid, lucene_docid, tf, and bm25_weight] of the term in document identified by docid.
        """
        termstats = defaultdict(list)  # Maps from "term" -> list[TermStat]
        if from_relevant_docs:
            if query.qid in self.query_rel_docs_map:
                docids = self.query_rel_docs_map[query.qid]
            # if there are no relevant docs for the query, then return empty dictionary
            else:
                return termstats
        else:
            if query.qid in self.query_non_rel_docs_map:
                docids = self.query_non_rel_docs_map[query.qid]
            # if there are no non-relevant docs for the query, then return empty dictionary
            else:
                return termstats
        print(f"No. of docids to traverse: {len(docids)}")
        for docid, lucene_docid in docids:
            tvec = self.termvectors.get(lucene_docid, CONTENTS_FIELD)
            doc_len = tvec.getSumTotalTermFreq()
            tvec_iter = tvec.iterator()
            for term in BytesRefIterator.cast_(tvec_iter):
                term_str = term.utf8ToString()
                tf = tvec_iter.totalTermFreq()
                weight = self._compute_bm25_weight(
                    term_str, tf, doc_len, is_query_term=False
                )
                termstats[term_str].append(
                    SimpleNamespace(
                        docid=docid, lucene_docid=lucene_docid, tf=tf, weight=weight
                    )
                )
        tqdm.write(f"No. of terms collected: {len(termstats)}")
        return termstats

    def compute_rocchio_vector(
        self,
        query: TRECQuery,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> QueryVector:
        query_terms = set(self._get_query_terms(query))
        query_termstats_rel = self._get_termstats(query, from_relevant_docs=True)
        query_termstats_non_rel = self._get_termstats(query, from_relevant_docs=False)
        rocchio_vector = QueryVector()
        for term, rel_stats in query_termstats_rel.items():
            if term in query_terms:
                weight = alpha * self._compute_bm25_weight(
                    term, 1, len(query_terms), True
                )
                query_terms.remove(term)
            else:
                weight = 0.0
            rel_avg_weight = sum(stat.weight for stat in rel_stats) / max(
                1, len(query_termstats_rel)
            )
            non_rel_stats = query_termstats_non_rel[term]
            non_rel_avg_weight = sum(stat.weight for stat in non_rel_stats) / max(
                1, len(query_termstats_non_rel)
            )
            weight += beta * rel_avg_weight - gamma * non_rel_avg_weight
            if weight < 0:  # don't add term to rocchio vector if weight < 0
                continue
            rocchio_vector[term] = SimpleNamespace(
                weight=weight, rel_docs_freq=len(query_termstats_rel[term])
            )
        for term in query_terms:
            weight = alpha * self._compute_bm25_weight(term, 1, len(query_terms), True)
            if term in query_termstats_non_rel:
                non_rel_stats = query_termstats_non_rel[term]
                non_rel_avg_weight = sum(stat.weight for stat in non_rel_stats) / max(
                    1, len(query_termstats_non_rel)
                )
                weight -= gamma * non_rel_avg_weight
                if weight < 0:  # dont't add term if weight < 0
                    continue
            rocchio_vector[term] = SimpleNamespace(
                weight=weight, rel_docs_freq=len(query_termstats_rel[term])
            )
        return rocchio_vector

    def tweak_query_vector(
        self,
        query: TRECQuery,
        query_vector: QueryVector,
        tweak_magnitude_list: list[float] = [4.0, 2.0, 1.0, 0.5, 0.25],
    ) -> QueryVector:
        current_map, _ = self.computeAP(query.qid, query_vector)
        if current_map is None:  # if relevance not present in qrel, nothing to do
            return query_vector
        for mag in tweak_magnitude_list:
            for term, stat in query_vector.vector.items():
                current_weight = stat.weight
                tqdm.write(
                    f"Term: {term:20s}Current Weight: {current_weight:.3f}, Current AP: {current_map:.3f}, Tweak Magnitude: {mag}"
                )
                nudged_weight = (1 + mag) * current_weight
                query_vector[term].weight = nudged_weight
                nudged_map, _ = self.computeAP(query.qid, query_vector)
                if nudged_map >= current_map:
                    tqdm.write(
                        f"Term: {term:20s}Nudged  Weight: {nudged_weight:.3f}, Nudged  AP: {nudged_map:.3f} -- Keeping   Nudged Weight"
                    )
                    current_map = nudged_map
                    if current_map == 1.0:  # stop computing if MAP is already 1
                        return query_vector
                else:
                    tqdm.write(
                        f"Term: {term:20s}Nudged  Weight: {nudged_weight:.3f}, Nudged  AP: {nudged_map:.3f} -- Reverting Nudged Weight"
                    )
                    query_vector[term].weight = current_weight
        return query_vector

    def generate(
        self,
        extracted_queries_path: str,
        weights_store_path: str,
        run_store_path: str,
        alpha=2.0,
        beta=64.0,
        gamma=64.0,
        runid="ideal_query",
        num_expansion_terms=200,
        num_top_docs=1000,
        tweak_magnitude_list=[4.0, 2.0, 1.0, 0.5, 0.25],
    ):
        query_reader = csv.reader(open(extracted_queries_path, "r"), delimiter="\t")
        for row in query_reader:
            qid = row[0]
            query_text = row[1]

            query = TRECQuery(qid, query_text)

            ## STEP 1: Compute initial rocchio vector
            print("Computing Rocchio Vector...")
            query_rocchio_vector = self.compute_rocchio_vector(
                query, alpha, beta, gamma
            )

            ## STEP 2: Remove rare terms
            print("Removing rare terms...")
            # Sort it according to decreasing order of frequency in relevant documents
            query_rocchio_vector.sort(lambda stat: stat.rel_docs_freq)
            num_valid_terms = 0
            min_threshold = 0.02 * len(self.query_rel_docs_map[query.qid])
            for term, stat in query_rocchio_vector.vector.items():
                if stat.rel_docs_freq < min_threshold:
                    break
                num_valid_terms += 1
            query_rocchio_vector.trim(num_valid_terms)

            ## STEP 3: Sort it according to decreasing order of weights
            query_rocchio_vector.sort()

            ## STEP 4: Trimming rocchio to terms with top num_expansion_terms weights
            print(f"Trimming Rocchio Vector to top {num_expansion_terms} terms.")
            query_rocchio_vector.trim(num_expansion_terms)

            ## STEP 5: Start tweaking
            query_rocchio_vector = self.tweak_query_vector(
                query,
                query_rocchio_vector,
                tweak_magnitude_list=tweak_magnitude_list,
            )

            ## STEP 6: Compute final AP and final run
            final_ap, final_run = self.computeAP(
                query.qid, query_rocchio_vector, num_top_docs
            )
            print(f"Final MAP: {final_ap:.3f}")

            ## STEP 7: Store run of final tweaked query
            store_run(
                final_run,
                run_output_path=run_store_path,
                runid=runid,
                append=True,
            )

            ## STEP 8: Store expanded query
            query_rocchio_vector.store(query.qid, weights_store_path, append=True)


if __name__ == "__main__":
    index_path = TREC_INDEX_DIR_PATH
    stopwords_path = STOPWORDS_FILE_PATH
    actual_qrel_path = TREC_QREL_FILE_PATH
    restrict_qrel_path = f"{ROOT_DIR}/qrels/bm25_intersect_trec678rb.qrel"

    iqg = IdealQueryGeneration(
        index_path, stopwords_path, actual_qrel_path, restrict_qrel_path
    )

    # Rocchio Vector hyperparams
    alpha = 2.0
    beta = 64.0
    gamma = 64.0

    # trim rocchio vector to top NUM_EXPANSION_TERMS according to weights
    num_expansion_terms = 200
    # number of retrived documents for computing AP
    num_top_docs = 1000

    # Tweak magnitude list
    tweak_magnitude_list = [4.0, 2.0, 1.0, 0.5, 0.25]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "extracted_queries_path", help="Path to extracted queries", type=str
    )
    parser.add_argument("--runid", help="Run ID", type=str, default="ideal_query")
    parser.add_argument(
        "--split",
        help="If queries are split for parallization",
        action="store_true",
    )
    args = parser.parse_args()

    basename = os.path.basename(args.extracted_queries_path)

    # Weights will be stored in ideal-queries/trec678/weights/runid.term_weights file
    # or ideal-queries/trec678/weights/runid-split/{query_file_basename}.term_weights files if split
    weights_store_path = f"{ROOT_DIR}/ideal-queries/trec678/weights/{args.runid}{f"-split/{basename}" if args.split else ""}.term_weights"
    # Runs will be stored in ideal-queries/trec678/runs/runid.run file
    # or ideal-queries/trec678/runs/runid-split/{query_file_basename}.run files if split
    run_store_path = f"{ROOT_DIR}/ideal-queries/trec678/runs/{args.runid}{f"-split/{basename}" if args.split else ""}.run"

    # Don't compute if weights or run file already present
    if os.path.exists(weights_store_path) or os.path.exists(run_store_path):
        print("Run file or Term weight file already exists!")
        exit(1)

    iqg.generate(
        args.extracted_queries_path,
        weights_store_path,
        run_store_path,
        alpha,
        beta,
        gamma,
        args.runid,
        num_expansion_terms,
        num_top_docs,
        tweak_magnitude_list=tweak_magnitude_list,
    )

    iqg.close()
