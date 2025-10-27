import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import chain
import os
import math

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
    PyTorch-based version of IdealQueryGeneration2
    Replaces sklearn LogisticRegression with a PyTorch linear regression version
    minimizing ||Xw - y||^2 where y ∈ {0, 100}.
    """

    def train_model(
        self,
        X,
        y,
        lr=0.1,
        epochs=5000,
        warmup_epochs=1000,
        l2_lambda=0.5,
        intercept=0.0,
    ):
        """
        Trains a linear regression model (least squares) using PyTorch.
        Returns an sklearn-compatible wrapper with coef_, intercept_, and predict().
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reformat y: 1 → 100, 0 → 0
        y_scaled = np.where(np.array(y) == 1, 100.0, 0.0)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1).to(device)

        model = nn.Linear(X.shape[1], 1, bias=False).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            [
                {"params": model.weight, "weight_decay": l2_lambda},
            ],
            lr=lr,
        )

        # Warmup + Cosine Annealing LR Scheduler
        def get_lr(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                decay_epoch = epoch - warmup_epochs
                total_decay_epochs = epochs - warmup_epochs
                return 1e-5 + 0.5 * (lr - 1e-5) * (
                    1 + math.cos(math.pi * decay_epoch / total_decay_epochs)
                )

        for epoch in range(epochs):
            optimizer.zero_grad()

            preds = model(X_tensor) - intercept
            loss = criterion(preds, y_tensor)

            loss.backward()
            optimizer.step()

            # Update learning rate manually
            lr_now = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(
                    f"[Torch] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f} - LR: {lr_now:.6f}"
                )

        with torch.no_grad():
            coef_ = model.weight.detach().cpu().numpy().flatten()

        # sklearn-like wrapper
        class TorchLinearRegWrapper:
            def __init__(self, coef_, intercept):
                self.coef_ = np.array([coef_])
                self.intercept_ = intercept

            def predict(self, X):
                X = np.array(X)
                return np.dot(X, self.coef_.T) - self.intercept_

        return TorchLinearRegWrapper(coef_, intercept)


if __name__ == "__main__":
    # Configuration
    index_dir = TREC_INDEX_DIR_PATH
    stopwords_path = STOPWORDS_FILE_PATH
    qrel_path = TREC_QREL_FILE_PATH

    iqg_torch = IdealQueryGenerationTorch2(index_dir, stopwords_path, qrel_path)
    qid_range = chain(range(301, 451), range(601, 701))

    #####################################
    ideal_q_runid = "ideal_query_torch_linear_rocchiotrunc"
    num_after_chi2_terms = None
    var_threshold = 0
    num_save_terms = None
    num_after_rocchio_terms = 1000
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
