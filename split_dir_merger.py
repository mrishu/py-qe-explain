import os
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("split_dir", help="Split Directory", type=str)
parser.add_argument("merged_file", help="Final Merged File", type=str)
args = parser.parse_args()

writer = csv.writer(open(args.merged_file, "w"), delimiter="\t")

for file in sorted(os.listdir(args.split_dir)):
    with open(args.split_dir + "/" + file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            writer.writerow(row)
