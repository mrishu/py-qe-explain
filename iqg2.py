import os
import csv
from os.path import exists
import pickle
import numpy as np
from collections import defaultdict
from numpy.random import weibull
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    VarianceThreshold,
)
from itertools import chain
import pytrec_eval
import math
from types import SimpleNamespace

from definitions import (
    ROOT_DIR,
    CONTENTS_FIELD,
    TREC_INDEX_DIR_PATH,
    TREC_QREL_FILE_PATH,
    STOPWORDS_FILE_PATH,
)
from utils import parse_queries, store_run, store_ap, store_mean_ap
from searcher import SearchAndEval
from classes import QueryVector

from org.apache.lucene.index import Term
from org.apache.lucene.util import BytesRefIterator


class IdealQueryGeneration2(SearchAndEval):
    def __init__(self, index_dir: str, stopwords_path: str, qrel_path: str) -> None:
        super().__init__(index_dir, stopwords_path, qrel_path)
        self.termvectors = self.reader.termVectors()
        self.no_of_docs = self.reader.numDocs()
        self.avg_dl = self.reader.getSumTotalTermFreq(CONTENTS_FIELD) / self.no_of_docs
        self.query_rel_docs_map = defaultdict(set)
        self.query_non_rel_docs_map = defaultdict(set)
        self.qrel_path = qrel_path
        with open(self.qrel_path, "r") as qrel_file:
            self.qrel = pytrec_eval.parse_qrel(qrel_file)
        self.read_qrel()

    def read_qrel(self) -> None:
        pickle_dir = self.qrel_path + "-pickles"
        os.makedirs(pickle_dir, exist_ok=True)
        if os.path.exists(os.path.join(pickle_dir, "query-docs-map.pkl")):
            with open(
                os.path.join(pickle_dir, "query-docs-map.pkl"), "rb"
            ) as pickle_file:
                self.query_rel_docs_map, self.query_non_rel_docs_map = pickle.load(
                    pickle_file
                )
            return
        for qid, docid_rel_map in self.qrel.items():
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

    def compute_bm25_weight(
        self, term: str, tf: float, doc_len: float, is_query_term: bool = False
    ) -> float:
        k1 = 1.2
        b = 0.75
        doc_freq = self.reader.docFreq(Term(CONTENTS_FIELD, term))
        idf = math.log(1 + (self.no_of_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        if is_query_term:
            return idf
        return idf * tf / (tf + k1 * (1 - b + b * doc_len / self.avg_dl))

    def get_docvec_dict(self, qid: str, from_relevant_docs: bool):
        pickles_dir = self.qrel_path + "-pickles"
        docvec_dict_dir = os.path.join(pickles_dir, "docvec_dicts")
        os.makedirs(docvec_dict_dir, exist_ok=True)

        docvec_dict = defaultdict(dict)
        terms = set()
        if from_relevant_docs:
            if qid in self.query_rel_docs_map:
                docids = self.query_rel_docs_map[qid]
            else:
                return docvec_dict, terms
        else:
            if qid in self.query_non_rel_docs_map:
                docids = self.query_non_rel_docs_map[qid]
            else:
                return docvec_dict, terms

        if from_relevant_docs:
            docvec_dict_file = os.path.join(docvec_dict_dir, f"{qid}-rel.pkl")
        else:
            docvec_dict_file = os.path.join(docvec_dict_dir, f"{qid}-non-rel.pkl")

        if os.path.exists(docvec_dict_file):
            with open(
                docvec_dict_file,
                "rb",
            ) as pickle_file:
                docvec_dict, terms = pickle.load(pickle_file)
        else:
            for docid, lucene_docid in docids:
                tvec = self.termvectors.get(lucene_docid, CONTENTS_FIELD)
                doc_len = tvec.getSumTotalTermFreq()
                tvec_iter = tvec.iterator()
                for term in BytesRefIterator.cast_(tvec_iter):
                    term_str = term.utf8ToString()
                    terms.add(term_str)
                    tf = tvec_iter.totalTermFreq()
                    weight = self.compute_bm25_weight(
                        term_str, tf, doc_len, is_query_term=False
                    )
                    docvec_dict[docid][term_str] = weight
            with open(
                docvec_dict_file,
                "wb",
            ) as pickle_file:
                pickle.dump((docvec_dict, terms), pickle_file)
        return docvec_dict, terms

    def convert_to_array(self, docvec_dict, term_index):
        docids, vectors = [], []
        for docid, term_weights in docvec_dict.items():
            vector = np.zeros(len(term_index))
            for term, weight in term_weights.items():
                idx = term_index.get(term)
                if idx is not None:
                    vector[idx] = weight
            docids.append(docid)
            vectors.append(vector)
        return docids, np.vstack(vectors)

    # This forms a 1D array of the query of ID qid
    # whose term_weights file is at query_path.
    # It is projected on terms in term_list.
    def form_query_1darray(self, qid, query_path, term_list):
        queries = parse_queries(query_path)
        qvec = queries[qid]
        final_vec = np.zeros(len(term_list))
        for i in range(len(term_list)):
            term = term_list[i]
            if term in qvec:
                final_vec[i] = qvec[term].weight
            else:
                final_vec[i] = 0.0
        return final_vec

    def get_QueryVector_from_1darray(self, array: np.ndarray, term_list: list):
        vector = QueryVector()
        for weight, term in zip(array, term_list):
            vector[term] = SimpleNamespace(weight=weight)
        return vector

    def select_features(self, X, y, term_list, num_after_chi2_terms, var_threshold):
        # Step 1: Eliminate low-variance features
        print("Removing low-variance features:", end=" ")
        var_thresh = VarianceThreshold(threshold=var_threshold)
        X_var = var_thresh.fit_transform(X)
        var_support = var_thresh.get_support(indices=True)
        term_list_var = [term_list[i] for i in var_support]
        print(f"Remaining {len(term_list_var)} terms")

        # Step 2: Select top-K features using the specified method
        print(
            f"Running feature selection using chi2: Selecting {num_after_chi2_terms} terms"
        )
        selector = SelectKBest(
            score_func=chi2, k=min(num_after_chi2_terms, X_var.shape[1])
        )
        X_selected = selector.fit_transform(X_var, y)
        selected_indices = selector.get_support(indices=True)
        scores = selector.scores_[selected_indices]
        sorted_idx = selected_indices[np.argsort(scores)[::-1]]

        # Map back to the original term list
        final_terms = [term_list_var[i] for i in sorted_idx]
        return X_selected[:, np.argsort(scores)[::-1]], final_terms

    def train_model(self, X, y):
        model = LogisticRegression(penalty="l2", solver="liblinear")
        model.fit(X, y)
        return model

    def process_qid(
        self,
        qid,
        num_after_chi2_terms,
        var_threshold,
        runid,
        num_save_terms,
        weight_save_file,
        run_save_file,
        ap_save_file,
    ):
        print("Processing QID:", qid)

        print("Forming Document vectors...")
        rel_docvec_dict, rel_terms = self.get_docvec_dict(qid, from_relevant_docs=True)
        non_rel_docvec_dict, nonrel_terms = self.get_docvec_dict(
            qid, from_relevant_docs=False
        )
        terms = rel_terms | nonrel_terms
        term_list = sorted(terms)

        term_index = {term: idx for idx, term in enumerate(term_list)}
        rel_ids, rel_vecs = self.convert_to_array(rel_docvec_dict, term_index)
        nonrel_ids, nonrel_vecs = self.convert_to_array(non_rel_docvec_dict, term_index)

        X = np.vstack([rel_vecs, nonrel_vecs])
        y = np.zeros(len(rel_ids) + len(nonrel_ids))
        y[: len(rel_ids)] = 1

        print(f"#terms: {len(term_list)}")
        print(f"#rel docs: {len(rel_ids)}, #non-rel docs: {len(nonrel_ids)}")

        # Feature Selection: chi2 then mutual_info
        X, term_list = self.select_features(
            X, y, term_list, num_after_chi2_terms, var_threshold
        )

        # Train Logistic Regression
        model = self.train_model(X, y)
        coef = model.coef_[0]
        print("Coeffiencent norm squared:", (np.linalg.norm(model.coef_[0])) ** 2)
        print("Intercept:", model.intercept_)

        print(
            "Class Probabilites of model coefficients:",
            model.predict_proba(coef.reshape(1, -1)),
        )

        # Original ideal query class probabilities
        # orig_ideal_query_path = (
        #     "./ideal-queries/trec678/weights/ideal_query.term_weights"
        # )
        # orig_ideal_query_arr = self.form_query_1darray(
        #     qid, orig_ideal_query_path, term_list
        # )
        # print(
        #     "Class Probabilities of original ideal query:",
        #     model.predict_proba(orig_ideal_query_arr.reshape(1, -1)),
        # )

        # Compute run and AP
        print("Computing AP and run...", end=" ")
        qvec = self.get_QueryVector_from_1darray(coef, term_list)
        qvec.sort_by_stat()
        if num_save_terms is not None:
            qvec.trim(num_save_terms)
        ap, run = self.computeAP_and_run(qid, qvec, num_top_docs=1000)
        print(f"AP acheived: {ap}")

        # Save run, ap, weights
        print("Saving AP, run and weights...", end=" ")
        ap_dict = {qid: ap}
        store_run(run, run_save_file, runid, append=True)
        store_ap(ap_dict, ap_save_file, append=True)
        qvec.store_txt(qid, weight_save_file, append=True)
        print("Done!")
        print()

    def run(
        self,
        qids,
        num_after_chi2_terms,
        var_thershold,
        runid,
        num_save_terms,
        weights_save_file,
        run_save_file,
        ap_save_file,
    ):
        for qid in qids:
            qid = str(qid)
            if qid not in self.qrel:
                continue
            self.process_qid(
                qid,
                num_after_chi2_terms,
                var_thershold,
                runid,
                num_save_terms,
                weights_save_file,
                run_save_file,
                ap_save_file,
            )


if __name__ == "__main__":
    index_dir = TREC_INDEX_DIR_PATH
    stopwords_path = STOPWORDS_FILE_PATH
    qrel_path = TREC_QREL_FILE_PATH

    iqg2 = IdealQueryGeneration2(index_dir, stopwords_path, qrel_path)
    qid_range = chain(range(301, 451), range(601, 701))

    #####################################
    ideal_q_runid = "ideal_query_chi2_lr"
    num_after_chi2_terms = 10000
    var_thershold = 1e-4
    num_save_terms = 1000
    # num_save_terms = None  # set to None to save all terms
    #####################################

    ideal_query_path = os.path.join(ROOT_DIR, "ideal-queries", "trec678")

    weight_save_file = os.path.join(
        ideal_query_path, "weights", f"{ideal_q_runid}.term_weights"
    )
    run_save_file = os.path.join(ideal_query_path, "runs", f"{ideal_q_runid}.run")
    ap_save_file = os.path.join(ideal_query_path, "aps", f"{ideal_q_runid}.ap")

    if (
        os.path.exists(weight_save_file)
        or os.path.exists(run_save_file)
        or os.path.exists(ap_save_file)
    ):
        inp = input(
            f"Either of weights, run or ap files of ideal query {ideal_q_runid} exists. Overwrite (Y/N): "
        )
        if inp == "y" or inp == "Y":
            if os.path.exists(weight_save_file):
                os.remove(weight_save_file)
            if os.path.exists(run_save_file):
                os.remove(run_save_file)
            if os.path.exists(ap_save_file):
                os.remove(ap_save_file)
        else:
            exit(1)

    iqg2.run(
        qid_range,
        num_after_chi2_terms,
        var_thershold,
        ideal_q_runid,
        num_save_terms,
        weight_save_file,
        run_save_file,
        ap_save_file,
    )
    store_mean_ap(ap_save_file)
