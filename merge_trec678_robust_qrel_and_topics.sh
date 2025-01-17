cat ./trec678rb/qrels/trec678_301-450.qrel ./trec678rb/qrels/robust_601-700.qrel >./trec678rb/qrels/trec678rb.qrel
cat ./trec678rb/topics/trec678.xml ./trec678rb/topics/robust.xml >./trec678rb/topics/trec678rb.xml
sed -i '3027d;3028d' ./trec678rb/topics/trec678rb.xml
