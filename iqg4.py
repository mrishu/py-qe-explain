import torch
import numpy as np
from itertools import chain
import os

from iqg2 import IdealQueryGeneration2
from definitions import (
    ROOT_DIR,
    TREC_INDEX_DIR_PATH,
    TREC_QREL_FILE_PATH,
    STOPWORDS_FILE_PATH,
)
from utils import store_mean_ap


class IdealQueryGenerationTorch2(IdealQueryGeneration2):
    """
    Ridge regression with Cholesky decomposition.
    Solves: min_w ||Xw - y||^2 + λ||w||^2
    Uses GPU if available, CPU otherwise.
    """

    def ridge_regression_cholesky_gpu(self, X, y, l2_lambda=0.5):
        """
        Solve w = (X^T X + λ I)^(-1) X^T y using Cholesky (GPU if available).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_t = torch.tensor(X, dtype=torch.float64, device=device)
        y_t = torch.tensor(y, dtype=torch.float64, device=device).view(-1, 1)

        n_features = X_t.shape[1]
        A = X_t.T @ X_t + l2_lambda * torch.eye(
            n_features, dtype=torch.float64, device=device
        )
        b = X_t.T @ y_t

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(A)
            w = torch.cholesky_solve(b, L)
        except RuntimeError:
            # Fallback: use solve if Cholesky fails
            print("[WARN] Cholesky failed, falling back to torch.linalg.solve")
            w = torch.linalg.solve(A, b)

        return w.detach().cpu().numpy().flatten()

    def train_model(self, X, y, l2_lambda=0.1, intercept=0.0):
        # Map y: 1 -> 100, 0 -> 0
        y_scaled = np.where(np.array(y) == 1, 100.0, 0.0)

        coef_ = self.ridge_regression_cholesky_gpu(X, y_scaled, l2_lambda)

        # sklearn-like wrapper
        class TorchLinearRegWrapper:
            def __init__(self, coef_, intercept):
                self.coef_ = np.array([coef_])
                self.intercept_ = intercept

            def predict(self, X):
                X = np.array(X)
                return np.dot(X, self.coef_.T) + self.intercept_

        return TorchLinearRegWrapper(coef_, intercept)


if __name__ == "__main__":
    # Configuration
    index_dir = TREC_INDEX_DIR_PATH
    stopwords_path = STOPWORDS_FILE_PATH
    qrel_path = TREC_QREL_FILE_PATH

    iqg_torch = IdealQueryGenerationTorch2(index_dir, stopwords_path, qrel_path)
    qid_range = chain(range(301, 451), range(601, 701))

    #####################################
    ideal_q_runid = "ideal_query_linear_direct"
    num_after_chi2_terms = 10000
    num_after_rocchio_terms = None
    var_threshold = 1e-4
    num_save_terms = 1000
    l2_lambda = 0.1
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
        if inp.lower() == "y":
            for f in [weight_save_file, run_save_file, ap_save_file]:
                if os.path.exists(f):
                    os.remove(f)
        else:
            exit(1)

    iqg_torch.run(
        qid_range,
        num_after_chi2_terms,
        num_after_rocchio_terms,
        var_threshold,
        ideal_q_runid,
        num_save_terms,
        weight_save_file,
        run_save_file,
        ap_save_file,
    )

    store_mean_ap(ap_save_file)
