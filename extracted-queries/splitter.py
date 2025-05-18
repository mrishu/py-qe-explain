import os
import csv
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "file_path", help="extracted queries file path to split", type=str
)
args = argparser.parse_args()

basename = os.path.basename(args.input)

reader = csv.reader(open(args.input, "r"), delimiter="\t")
dir = os.path.dirname(os.path.abspath(args.input))

os.makedirs(os.path.join(dir, f"{basename}-split"), exist_ok=True)

for row in reader:
    writer = csv.writer(
        open(os.path.join(dir, f"{basename}-split", row[0]), "w"), delimiter="\t"
    )
    writer.writerow(row)
