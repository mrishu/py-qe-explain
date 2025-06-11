from iqg import IdealQueryGeneration
from definitions import (
    TREC_INDEX_DIR_PATH,
    STOPWORDS_FILE_PATH,
    TREC_QREL_FILE_PATH,
)
from utils import parse_queries, store_ap, store_run, store_mean_ap

########################################
query_weight_file = "./ideal-queries/trec678/weights/ideal_query_chi2_lr.term_weights"
runid = "ideal_query_chi2_lr_pruned"
tweak_magnitude_list = [-1.0]

# Store path
weights_store_path = (
    "./ideal-queries/trec678/weights/ideal_query_chi2_lr_pruned.term_weights"
)
run_store_path = "./ideal-queries/trec678/runs/ideal_query_chi2_lr_pruned.run"
ap_store_path = "./ideal-queries/trec678/aps/ideal_query_chi2_lr_pruned.ap"

# Trimming
num_terms = 200
# num_terms = None  # for no trimming
########################################

index_path = TREC_INDEX_DIR_PATH
stopwords_path = STOPWORDS_FILE_PATH
actual_qrel_path = TREC_QREL_FILE_PATH

iqg = IdealQueryGeneration(
    index_path, stopwords_path, actual_qrel_path, restrict_qrel_path=None
)

query_vectors = parse_queries(query_weight_file)

for qid, qvec in query_vectors.items():
    print("############# Processing Query:", qid, "#####################")

    qvec.remove_non_positive_weights()
    qvec.sort_by_stat()
    if num_terms is not None:
        qvec.trim(num_terms)

    tweaked_qvec = iqg.tweak_query_vector(
        str(qid), qvec, tweak_magnitude_list, num_top_docs=1000, runid=runid
    )

    qvec.remove_zero_weights()  # remove zero weights before storing

    final_ap, final_run = iqg.computeAP_and_run(qid, tweaked_qvec, num_top_docs=1000)

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

    qvec.store_txt(qid, weights_store_path, append=True)

store_mean_ap(ap_store_path)
