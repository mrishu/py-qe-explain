import xml.etree.ElementTree as ET
import csv

from tqdm import tqdm

from definitions import TREC_QUERIES_FILE_PATH, TREC_EXTRACTED_QUERIES_FILE


writer = csv.writer(open(TREC_EXTRACTED_QUERIES_FILE, "w"), delimiter="\t")

trec_topics = ET.parse(TREC_QUERIES_FILE_PATH).getroot()
for top in tqdm(trec_topics, desc="Queries Processed"):
    qid = top[0].text.strip()  # this is query number
    query_text = top[1].text.strip()  # this will be our query
    narr = top[2].text.strip()
    writer.writerow([qid, query_text])
