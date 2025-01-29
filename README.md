# py-qe-explain
Query Expansion Explanation

## Dataset Link
[Link](https://drive.google.com/file/d/19qkzLYnz7NiE4KeqO9ZQ2YGtSB9QBcL1/view?usp=sharing)

## Dependencies
Apart from [pylucene-10.0.0](https://dlcdn.apache.org/lucene/pylucene/), the dependencies are:
1. [pytrec_eval](https://github.com/cvangysel/pytrec_eval)
2. tqdm (For progress bars)
```bash
pip install pytrec_eval tqdm
```


## Some preliminary results
- When restricted to queries 301-320 in `trec678`, the results for **`iqg.py`** are:
```
runid                 	all	idealQuery
num_q                 	all	20
num_ret               	all	20000
num_rel               	all	2050
num_rel_ret           	all	1821
map                   	all	0.8263
gm_map                	all	0.7996
Rprec                 	all	0.7853
bpref                 	all	0.8539
recip_rank            	all	1.0000
iprec_at_recall_0.00  	all	1.0000
iprec_at_recall_0.10  	all	0.9873
iprec_at_recall_0.20  	all	0.9790
iprec_at_recall_0.30  	all	0.9499
iprec_at_recall_0.40  	all	0.9092
iprec_at_recall_0.50  	all	0.8787
iprec_at_recall_0.60  	all	0.8369
iprec_at_recall_0.70  	all	0.7982
iprec_at_recall_0.80  	all	0.7226
iprec_at_recall_0.90  	all	0.5684
iprec_at_recall_1.00  	all	0.3439
P_5                   	all	0.9600
P_10                  	all	0.9050
P_15                  	all	0.8400
P_20                  	all	0.7850
P_30                  	all	0.7150
P_100                 	all	0.4560
P_200                 	all	0.3138
P_500                 	all	0.1634
P_1000                	all	0.0910
```

- When restricted to queries 301-320 in `trec678`, the results for **`iqg2.py`** are:
```
runid                 	all	idealQuery
num_q                 	all	20
num_ret               	all	20000
num_rel               	all	2050
num_rel_ret           	all	1820
map                   	all	0.8354
gm_map                	all	0.8065
Rprec                 	all	0.8026
bpref                 	all	0.8615
recip_rank            	all	1.0000
iprec_at_recall_0.00  	all	1.0000
iprec_at_recall_0.10  	all	0.9884
iprec_at_recall_0.20  	all	0.9842
iprec_at_recall_0.30  	all	0.9582
iprec_at_recall_0.40  	all	0.9063
iprec_at_recall_0.50  	all	0.8818
iprec_at_recall_0.60  	all	0.8401
iprec_at_recall_0.70  	all	0.8083
iprec_at_recall_0.80  	all	0.7329
iprec_at_recall_0.90  	all	0.6018
iprec_at_recall_1.00  	all	0.3579
P_5                   	all	0.9600
P_10                  	all	0.9050
P_15                  	all	0.8300
P_20                  	all	0.7800
P_30                  	all	0.7083
P_100                 	all	0.4610
P_200                 	all	0.3145
P_500                 	all	0.1637
P_1000                	all	0.0910
```

# Steps to reproduce
1. Download the dataset from [here](https://drive.google.com/file/d/19qkzLYnz7NiE4KeqO9ZQ2YGtSB9QBcL1/view?usp=sharing), extract it and move it to a directory called `collections` inside the project.
```
mkdir collections
tar -xvzf trec678rb.tar.gz
mv trec678rb collections
```
2. Index the dataset using [indexer_trec678rb.py](indexer_trec678rb.py).
```
python indexer_trec678rb.py
```
3. Generate a run file of top 1000 documents by standard BM25 retrieval.
```
python searcher.py
```
4. Intersect the generated run file with the original qrel file to produce a qrel file containing only documents
that occurred in the top 1000 standard BM25 retrieval.
```
python intersect_run_with_qrel.py test-runs/bm25.run qrels/trec678.qrels qrels/bm25_intersect_trec678rb.qrel
```
5. Generating Ideal Query weights:

  1. Way 1 (without parallelization but clean output):
  ```
  python iqg.py extracted-queries/trec678 --runid ideal-query-restrict
  ```
  This produces a files called:
    - A run file: `ideal-queries/trec678/runs/ideal-query-restrict.run`
    - A term_weights file: `ideal-queries/trec678/weights/ideal-query-restrict.term_weights`

  2. Way 2 (with parallelization):
  ```
  ./parallel_ideal_query_computer 12
  ```
  where 12 can be replaced by number of parallel jobs.

  This produces two directories:
  - A run directory: `ideal-queries/trec678/runs/ideal-query-restrict-split/`
  - A term_weights directory: `ideal-queries/trec678/weights/ideal-query-restrict-split/`
  These split directories will contain runs and term_weights for each query respectively.
  
  These can then be merged:
  ```
  python split_dir_merger.py ideal-queries/trec678/runs/ideal-query-restrict-split ideal-queries/trec678/runs/ideal-query-restrict.run
  ```
  and 
  ```
  python split_dir_merger.py ideal-queries/trec678/weights/ideal-query-restrict-split ideal-queries/trec678/weights/ideal-query-restrict.weights
  ```
