import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

# File paths (edit as needed)
num_rel_docs_file = "./num_rel_docs.txt"
ap_files = [
    "./ideal-queries/trec678/aps/ideal_query_torch_linear.ap",
    "./ideal-queries/trec678/aps/ideal_query_chi2_lr.ap",
]  # List your AP files here


# Function to read AP values from a file
def read_ap_file(filepath):
    ap_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            _, qid, ap = line.strip().split()
            if qid == "all":
                continue
            ap_dict[qid] = float(ap)
    return ap_dict


# Function to read relevant document counts
def read_rel_docs_file(filepath):
    rel_docs_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            qid, num = line.strip().split()
            rel_docs_dict[qid] = int(num)
    return rel_docs_dict


# Load number of relevant documents
rel_docs_dict = read_rel_docs_file(num_rel_docs_file)

# Colors and markers for plotting
colors = ["blue", "red", "green", "purple", "orange", "brown", "pink"]
markers = ["o", "o", "^", "D", "v", "x", "*"]

# Plot setup
plt.figure(figsize=(10, 7))

# Plot each AP file
for idx, ap_file in enumerate(ap_files):
    ap_dict = read_ap_file(ap_file)

    x_vals = []
    y_vals = []
    for qid in ap_dict:
        if qid in rel_docs_dict and rel_docs_dict[qid] > 0:
            log_rel_docs = rel_docs_dict[qid]  # natural log
            x_vals.append(log_rel_docs)
            y_vals.append(ap_dict[qid])

    label = os.path.basename(ap_file)  # Use filename as label
    color = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]
    plt.scatter(x_vals, y_vals, label=label, color=color, marker=marker, alpha=0.7)

# Final plot formatting
plt.xlabel("log(Number of Relevant Documents)")
plt.ylabel("Average Precision (AP)")
plt.title("Scatter Plot of #Rel Docs vs AP")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
