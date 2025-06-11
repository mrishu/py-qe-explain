import matplotlib.pyplot as plt
from utils import parse_queries


def plot_relevance_vs_query_length(query_vectors, relevance_file_path, label, color):
    # Step 1: Read relevance info from file
    qid_to_reldoc_count = {}
    with open(relevance_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # Skip malformed lines
            qid, rel_count = parts
            qid_to_reldoc_count[qid] = int(rel_count)

    # Step 2: Prepare data for plotting
    x_rels = []
    y_terms = []
    for qid, term_weights in query_vectors.items():
        if qid in qid_to_reldoc_count:
            num_terms = len(term_weights)
            num_rels = qid_to_reldoc_count[qid]
            if num_rels == 0:
                continue
            x_rels.append(num_rels)
            y_terms.append(num_terms)

    # Step 3: Plot
    plt.scatter(x_rels, y_terms, alpha=0.7, color=color, label=label)


query_vectors1 = parse_queries(
    "./ideal-queries/trec678/weights/ideal_query_short.term_weights"
)
query_vectors2 = parse_queries(
    "./ideal-queries/trec678/weights/ideal_query_chi2_lr_pruned.term_weights"
)
numrel_file = "num_rel_docs.txt"

plt.figure(figsize=(8, 6))
plot_relevance_vs_query_length(
    query_vectors1, numrel_file, label="IEQ0Pruned", color="blue"
)
plot_relevance_vs_query_length(
    query_vectors2, numrel_file, label="IEQ1Pruned", color="red"
)

plt.xlabel("Number of relevant documents")
plt.ylabel("Number of terms in query")
plt.title("Query Length vs. Number of Relevant Documents")
plt.legend()
plt.grid(True)
plt.show()
