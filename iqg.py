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
import logging

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
from utils import store_run, store_ap
from searcher import SearchAndEval

## Lucene imports
from org.apache.lucene.index import Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.util import BytesRefIterator

"""Overall Procedure:
1. Construct `query_rel_docs_map`, `query_non_rel_docs_map` dictionaries using the function `_read_restrict_qrel()` function
    from the `restrict_qrel_path` file.
    - `query_rel_docs_map` is a dictionary mapping from `qid` -> `list[docid]` of relevant documents.
    - `query_non_rel_docs_map` is a dictionary mapping from `qid` -> `list[docid]` of non-relevant documents.
    These two dictionaries are computed for the whole `IdealQueryGeneration` instance once.
2. For each query, initial Rocchio vector is computed using the `compute_rocchio_vector()` function.
    - The Rocchio vector will essentially be a dictionary mapping from `term` -> `weight`. And this `weight` will be computed as:
      ```
       alpha * weight of `term` in query
       + beta * average of `term` weights in relevant documents
       - gamma * average of `term` weights in non-relevant documents,
      ```
      where `term`s ideally varies over all possible terms.
      (But since, weight < 0 is not allowed, we can safely ignore terms that don't occur in the query or the relevant documents.)
3. Inside `compute_rocchio_vector()` function:
    1. We first invoke the `_get_termstats()` function to get the "term statistics" from relevant/non-relevant documents.
        This function first collects all the terms occuring in the relevant/non-relevant documents and then 
        returns a dictionary "term" -> list[(docid, weight)]. Here `weight` is the BM25 weight of the `term` in the document identified by `docid`.
        These are collected as `query_termstats_rel` and `query_termstats_non_rel` dictionaries for relevant/non-relevant documents respectively.
    2. We then go over all `term`s in `query_termstats_rel` and compute the Rocchio vector `weight` of `term` by:
        ```
        rel_avg_weight = avg(weights in query_termstats_rel[term])
        non_rel_avg_weight = avg(weights in query_termstats_non_rel[term])
        rocchio_vector[term] = alpha * weight of term in query + beta * rel_avg_weight - gamma * non_rel_avg_weight
        ```
    3. This dictionary `rocchio_vector` is then returned.
4. After the `rocchio_vector` for a query is computed, we tweak the weights of each `term` to increase the MAP as much as possible.
    We select a `tweak_magnitude` from the `tweak_magnitude_list = [4.0, 2.0, 1.0, 0.5, 0.25]`, and for each `term` in the `rocchio_vector`,
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
        actual_qrel_path: str,  # this will be used for evaluation
        restrict_qrel_path: Union[
            str, None
        ] = None,  # this will be used for reading relevance information for building rocchio_vector
    ) -> None:
        super().__init__(index_dir, stopwords_path, actual_qrel_path)
        self.termvectors = self.reader.termVectors()
        self.no_of_docs = self.reader.numDocs()
        self.avg_dl = self.reader.getSumTotalTermFreq(CONTENTS_FIELD) / self.no_of_docs
        self.query_rel_docs_map = defaultdict(set)
        self.query_non_rel_docs_map = defaultdict(set)
        if restrict_qrel_path is None:
            self.restrict_qrel_path = actual_qrel_path
        else:
            self.restrict_qrel_path = restrict_qrel_path
        with open(self.restrict_qrel_path, "r") as qrel_file:
            self.restrict_qrel = pytrec_eval.parse_qrel(qrel_file)
        self._read_restrict_qrel()

    def _read_restrict_qrel(self) -> None:
        pickle_dir = self.restrict_qrel_path + "-pickles"
        os.makedirs(pickle_dir, exist_ok=True)
        if os.path.exists(os.path.join(pickle_dir, "query-docs-map.pkl")):
            with open(
                os.path.join(pickle_dir, "query-docs-map.pkl"), "rb"
            ) as pickle_file:
                self.query_rel_docs_map, self.query_non_rel_docs_map = pickle.load(
                    pickle_file
                )
            return
        for qid, docid_rel_map in self.restrict_qrel.items():
            for docid, rel in docid_rel_map.items():
                if rel != 0:
                    lucene_docid = self.get_lucene_docid(docid)
                    if lucene_docid != -1:
                        self.query_rel_docs_map[qid].add((docid, lucene_docid))
                else:
                    lucene_docid = self.get_lucene_docid(docid)
                    if lucene_docid != -1:
                        self.query_non_rel_docs_map[qid].add((docid, lucene_docid))
        with open(os.path.join(pickle_dir, "query-docs-map.pkl"), "wb") as pickle_file:
            pickle.dump(
                (self.query_rel_docs_map, self.query_non_rel_docs_map),
                pickle_file,
            )

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
        """Given a query, it collects the terms from relevant (if from_relevant_docs is True) / non-relevant documents.
        It then returns a dictionary: term (str) -> list[SimpleNamespace],
        where each SimpleNamespace has term statistics containing
        [docid, lucene_docid, tf, and bm25_weight] of the term in document identified by docid.
        """
        pickles_dir = self.restrict_qrel_path + "-pickles"
        termstats_dir = os.path.join(pickles_dir, "termstats")
        os.makedirs(termstats_dir, exist_ok=True)
        termstats = defaultdict(list)  # Maps from "term" -> list[TermStat]
        # if there is no relevance information for the query, then return empty dictionary
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
        print(f"No. of docids to traverse: {len(docids)}")
        if from_relevant_docs:
            termstats_file = os.path.join(termstats_dir, f"{query.qid}-rel.pkl")
        else:
            termstats_file = os.path.join(termstats_dir, f"{query.qid}-non-rel.pkl")
        if os.path.exists(termstats_file):
            with open(
                termstats_file,
                "rb",
            ) as pickle_file:
                termstats = pickle.load(pickle_file)
        else:
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
            with open(
                termstats_file,
                "wb",
            ) as pickle_file:
                pickle.dump(termstats, pickle_file)
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
            rel_total_weight = sum(stat.weight for stat in rel_stats)
            rel_avg_weight = rel_total_weight / max(1, len(query_termstats_rel))
            non_rel_stats = query_termstats_non_rel[term]
            non_rel_total_weight = sum(stat.weight for stat in non_rel_stats)
            non_rel_avg_weight = non_rel_total_weight / max(
                1, len(query_termstats_non_rel)
            )
            weight += beta * rel_avg_weight - gamma * non_rel_avg_weight
            if weight <= 0:  # don't add term to rocchio vector if weight <= 0
                continue
            rocchio_vector[term] = SimpleNamespace(
                weight=weight,
                doc_freq=self.reader.docFreq(Term(CONTENTS_FIELD, term)),
                rel_docs_freq=len(query_termstats_rel[term]),
                rel_total_weight=rel_total_weight,
                non_rel_docs_freq=len(query_termstats_non_rel[term]),
                non_rel_total_weight=non_rel_total_weight,
            )
        # remaining terms in query_terms that don't occur in any relevant document
        for term in query_terms:
            weight = alpha * self._compute_bm25_weight(term, 1, len(query_terms), True)
            # if the term occurs in non-relevant documents
            if term in query_termstats_non_rel:
                non_rel_stats = query_termstats_non_rel[term]
                non_rel_avg_weight = sum(stat.weight for stat in non_rel_stats) / max(
                    1, len(query_termstats_non_rel)
                )
                weight -= gamma * non_rel_avg_weight
                if weight <= 0:  # dont't add term if weight <= 0
                    continue
                rocchio_vector[term] = SimpleNamespace(
                    weight=weight,
                    doc_freq=self.reader.docFreq(Term(CONTENTS_FIELD, term)),
                    rel_docs_freq=0.0,
                    rel_total_weight=0.0,
                    non_rel_docs_freq=len(query_termstats_non_rel[term]),
                    non_rel_total_weight=sum(
                        stat.weight for stat in query_termstats_non_rel[term]
                    ),
                )
            else:
                rocchio_vector[term] = SimpleNamespace(
                    weight=weight,
                    doc_freq=self.reader.docFreq(Term(CONTENTS_FIELD, term)),
                    rel_docs_freq=0.0,
                    rel_total_weight=0.0,
                    non_rel_docs_freq=0.0,
                    non_rel_total_weight=0.0,
                )
        return rocchio_vector

    def tweak_query_vector(
        self,
        qid: str,
        query_vector: QueryVector,
        tweak_magnitude_list: list[float] = [4.0, 2.0, 1.0, 0.5, 0.25],
        runid=None,  # for logging
    ) -> QueryVector:
        if runid:
            os.makedirs(os.path.join(ROOT_DIR, "logs", runid), exist_ok=True)
            logging.basicConfig(
                filename=os.path.join(ROOT_DIR, "logs", runid, f"{qid}.log"),
                level=logging.DEBUG,
                format="%(message)s",
            )
        current_map, _ = self.computeAP_and_run(qid, query_vector)
        if (
            current_map is None
        ):  # if relevance information not present in qrel file, nothing to do
            return query_vector
        for mag in tweak_magnitude_list:
            for term, stat in query_vector.vector.items():
                current_weight = stat.weight
                nudged_weight = (1 + mag) * current_weight
                query_vector[term].weight = nudged_weight
                nudged_map, _ = self.computeAP_and_run(qid, query_vector)
                tqdm.write(
                    f"Term: {term:20s}Current Weight: {current_weight:.3f}, Current AP: {current_map:.3f}, Tweak Magnitude: {mag}"
                )
                assert nudged_map is not None
                if runid:
                    logging.info(
                        f"{qid}\t{mag}\t{term}\t{current_weight}\t{nudged_weight}\t{current_map}\t{nudged_map}"
                    )
                if nudged_map >= current_map:
                    tqdm.write(
                        f"Term: {term:20s}Nudged  Weight: {nudged_weight:.3f}, Nudged  AP: {nudged_map:.3f} -- Keeping   Nudged Weight"
                    )
                    current_map = nudged_map
                else:
                    query_vector[term].weight = current_weight
                    tqdm.write(
                        f"Term: {term:20s}Nudged  Weight: {nudged_weight:.3f}, Nudged  AP: {nudged_map:.3f} -- Reverting Nudged Weight"
                    )
        return query_vector

    def generate(
        self,
        extracted_queries_path: str,
        weights_store_path: str,
        ideal_query_pickle_dir: str,
        run_store_path: str,
        ap_store_path: str,
        alpha=2.0,
        beta=64.0,
        gamma=64.0,
        runid="ideal_query",
        num_expansion_terms=200,
        num_top_docs=1000,
        tweak_magnitude_list=[4.0, 2.0, 1.0, 0.5, 0.25],
    ):
        extracted_query_reader = csv.reader(
            open(extracted_queries_path, "r"), delimiter="\t"
        )
        for row in extracted_query_reader:
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
            query_rocchio_vector.sort_by_stat(lambda stat: stat.rel_docs_freq)
            num_valid_terms = 0
            min_threshold = 0.02 * len(self.query_rel_docs_map[qid])
            for _, stat in query_rocchio_vector.vector.items():
                if stat.rel_docs_freq < min_threshold:
                    break
                num_valid_terms += 1
            query_rocchio_vector.trim(num_valid_terms)

            ## STEP 3: Sort it according to decreasing order of weights
            query_rocchio_vector.sort_by_stat()

            ## STEP 4: Trimming rocchio to terms with top num_expansion_terms weights
            print(f"Trimming Rocchio Vector to top {num_expansion_terms} terms.")
            query_rocchio_vector.trim(num_expansion_terms)

            ## STEP 5: Start tweaking
            query_rocchio_vector = self.tweak_query_vector(
                qid,
                query_rocchio_vector,
                tweak_magnitude_list=tweak_magnitude_list,
                runid=runid,
            )

            ## STEP 6: Compute final AP and final run
            final_ap, final_run = self.computeAP_and_run(
                qid, query_rocchio_vector, num_top_docs
            )

            ## STEP 7: Store run of final tweaked query
            if final_ap is not None:
                print(f"Final MAP: {final_ap:.3f}")
                store_run(
                    final_run,
                    run_output_path=run_store_path,
                    runid=runid,
                    append=True,
                )
                aps = {qid: final_ap}
                store_ap(aps, ap_output_path=ap_store_path, append=True)

            ## STEP 8: Store tweaked query weights
            query_rocchio_vector.store_txt(qid, weights_store_path, append=True)

            ## STEP 9: Store tweaked query pickle
            query_rocchio_vector.store_pickle(
                os.path.join(ideal_query_pickle_dir, runid, f"{qid}.pkl")
            )


if __name__ == "__main__":
    index_path = TREC_INDEX_DIR_PATH
    stopwords_path = STOPWORDS_FILE_PATH
    actual_qrel_path = TREC_QREL_FILE_PATH
    # restrict_qrel_path = os.path.join(
    #     ROOT_DIR, "qrels", "bm25_intersect_trec678rb.qrel"
    # )
    restrict_qrel_path = None  # use actual_qrel_path as restrict_qrel_path

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

    # Weights will be stored in ideal-queries/trec678/weights/runid.term_weights file
    # or ideal-queries/trec678/weights/{runid}-split/{qid}.term_weights files if split
    # Runs will be stored in ideal-queries/trec678/runs/{runid}.run file
    # or ideal-queries/trec678/runs/{runid}-split/{qid}.run files if split
    ideal_query_base_dir = os.path.join(ROOT_DIR, "ideal-queries", "trec678")
    if args.split:
        qid = os.path.basename(args.extracted_queries_path)
        weights_store_path = os.path.join(
            ideal_query_base_dir,
            "weights",
            f"{args.runid}-split",
            f"{qid}.term_weights",
        )
        run_store_path = os.path.join(
            ideal_query_base_dir,
            "runs",
            f"{args.runid}-split",
            f"{qid}.run",
        )
    else:
        weights_store_path = os.path.join(
            ideal_query_base_dir,
            "weights",
            f"{args.runid}.term_weights",
        )
        run_store_path = os.path.join(ideal_query_base_dir, "runs", f"{args.runid}.run")
    ap_store_path = os.path.join(
        ideal_query_base_dir,
        "aps",
        f"{args.runid}.ap",
    )

    ideal_query_pickle_dir = os.path.join(ROOT_DIR, "ideal-queries-pickle", "trec678")

    # Don't compute if weights or run file already present
    if os.path.exists(weights_store_path) or os.path.exists(run_store_path):
        print("Run file or Term weight file already exists!")
        exit(1)

    iqg.generate(
        args.extracted_queries_path,
        weights_store_path,
        ideal_query_pickle_dir,
        run_store_path,
        ap_store_path,
        alpha,
        beta,
        gamma,
        args.runid,
        num_expansion_terms,
        num_top_docs,
        tweak_magnitude_list=tweak_magnitude_list,
    )

    iqg.close()
