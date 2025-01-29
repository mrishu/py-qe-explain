import os
import lucene

lucene.initVM()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

ID_FIELD = "id"
CONTENTS_FIELD = "contents"

TREC_COLL_DIR_PATH = os.path.join(ROOT_DIR, "collections", "trec678rb")
TREC_DOCS_DIR_PATH = os.path.join(TREC_COLL_DIR_PATH, "documents")
TREC_QUERIES_FILE_PATH = os.path.join(TREC_COLL_DIR_PATH, "topics" "trec678rb.xml")
TREC_INDEX_DIR_PATH = os.path.join(ROOT_DIR, "indexed", "trec678rb")
TREC_QREL_FILE_PATH = os.path.join(ROOT_DIR, "qrels", "trec678rb.qrel")
TREC_EXTRACTED_QUERIES_FILE = os.path.join(ROOT_DIR, "extracted-queries", "trec678rb")
TREC_EXTRACTED_QUERIES_SPLIT_DIR = os.path.join(
    ROOT_DIR, "extracted-queries", "trec678rb-split"
)

STOPWORDS_FILE_PATH = os.path.join(ROOT_DIR, "resources", "smart-stopwords")
