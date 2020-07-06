#!/bin/bash


if [ "$#" -lt 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_command.sh <NEO QUERY> <OUT FILE>"
    exit 1
fi

neo_query="$1"
out_name="$2"

. /root/env/bin/activate

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.252.b09-2.el7_8.x86_64/jre


DIR=$(dirname $0)

time spark-submit --num-executors=384 --driver-memory 12g --executor-memory 24g --jars /opt/sparkx/neo4j-spark-connector-full-2.4.5-M1.jar --conf spark.neo4j.user=neo4j --conf spark.neo4j.password=test --class com.navercorp.Main $DIR/target/node2vec-0.0.1-SNAPSHOT.jar --cmd neo2vec --dim 10 --p 100.0 --q 100.0 --walkLength 5 --output "$out_name" --input "kirokhayeh.bin" --neoQuery "$neo_query"

