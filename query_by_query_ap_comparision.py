from utils import parse_ap
import csv

ap_file1 = "./ideal-queries/trec678/aps/ideal_query.ap"
ap_file2 = "./ideal-queries/trec678/aps/ideal_query_chi2_lr.ap"
outfile = "./studies/qbyq_comparision.txt"
num_rel_file = "./studies/num_rel_docs.txt"

num_rel = {}
with open(num_rel_file, "r") as f:
    for line in f:
        qid, num = line.strip().split()
        num_rel[qid] = int(num)

aps1 = parse_ap(ap_file1)
aps2 = parse_ap(ap_file2)

# Display explicitly where the difference in APs is more than threshold
threshold = 0.2

with open(outfile, "w") as f:
    writer = csv.writer(f, delimiter="\t")
    for qid, ap1 in aps1.items():
        if qid not in aps2:
            continue
        ap2 = aps2[qid]
        if abs(ap1 - ap2) > threshold:
            print(qid, num_rel[qid], ap1, ap2, ap1 - ap2, sep="\t")
        writer.writerow([qid, ap1, ap2])
