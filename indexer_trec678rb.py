#!/usr/bin/env python3

""" -- INDEXING --

There are four types of files, starting with:
fb, fr, ft, la.
Each file contains lots of documents.

Each document is inside a <doc> tag which contains many
children tags which enclose different fields for each document.

We are indexing two fields from each document, namely: "id" and "contents".
1. "id" field is derived from <docno> tag in all four types of files.
2. "contents" field is derived from <text> tag in all four types of files.

- We are storing "id" as StringField (as it is not tokenized).
- We are NOT storing "contents" but indexing it as TextField (as it is tokenized),
    and we are also storing the TermVectors for this field.

NOTE:
- <docno> tag is present in ALL documents, but some dont contain <text> tag.
- We ignore documents with no <text> tag.
- We ignore information inside all other tags for all files.
"""

from definitions import (
    ID_FIELD,
    CONTENTS_FIELD,
    TREC_INDEX_DIR_PATH,
    STOPWORDS_FILE_PATH,
    TREC_DOCS_DIR_PATH,
)

import os
from bs4 import BeautifulSoup
import argparse

import lucene
from java.io import File
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.core import StopFilter
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search.similarities import BM25Similarity
import org.apache.lucene.document as document

from tqdm import tqdm


# Make sure index-dir directory is removed before re-indexing
indexPath = File(TREC_INDEX_DIR_PATH).toPath()
indexDir = FSDirectory.open(indexPath)
with open(STOPWORDS_FILE_PATH, "r") as stopwords_file:
    stopwords = stopwords_file.read().strip().split()
analyzer = EnglishAnalyzer(StopFilter.makeStopSet(stopwords))
writerConfig = IndexWriterConfig(analyzer)
writerConfig.setSimilarity(BM25Similarity())
writer = IndexWriter(indexDir, writerConfig)


# Make new IndexableFieldType that stores term vectors (we are doing this for TEXT field)
termvecstore_TextField = document.FieldType(document.TextField.TYPE_NOT_STORED)
termvecstore_TextField.setStoreTermVectors(True)


def indexDoc(docno, text):
    doc = document.Document()
    doc.add(document.Field(ID_FIELD, docno, document.StringField.TYPE_STORED))
    doc.add(document.Field(CONTENTS_FIELD, text, termvecstore_TextField))
    writer.addDocument(doc)


doc_count = 1
for filename in tqdm(os.listdir(TREC_DOCS_DIR_PATH)):
    with open(
        os.path.join(TREC_DOCS_DIR_PATH, filename), "r", encoding="ISO-8859-1"
    ) as fp:
        soup = BeautifulSoup(fp, "html.parser")
        doc = soup.find("doc")
        while doc is not None:
            docno = doc.findChildren("docno")[0].get_text().strip()
            text = doc.findChildren("text")
            # ignore document if no <text> tag present
            if len(text) == 0:
                doc = doc.find_next("doc")
                continue
            text = text[0].get_text().strip()
            # it can be that <text> tag is present but nothing is inside. ignore that too.
            if text == "":
                doc = doc.find_next("doc")
                continue
            # print(f"Indexing {doc_count} -- {docno}")
            indexDoc(docno, text)
            doc = doc.find_next("doc")
            doc_count += 1

writer.close()
