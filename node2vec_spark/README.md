# node2vec on spark

This library is a implementation using scala for running on spark of *node2vec* as described in the paper:
> node2vec: Scalable Feature Learning for Networks.
> Aditya Grover and Jure Leskovec.
> Knowledge Discovery and Data Mining, 2016.
> <Insert paper link>

The *node2vec* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. Please check the [project page](https://snap.stanford.edu/node2vec/) for more details. 


### Building node2vec_spark
**In order to build node2vec_spark, use the following:**

```
$ git clone https://github.com/Skarface-/node2vec.git
$ mvn clean package
```

**and requires:**<br/>
Maven 3.0.5 or newer<br/>
Java 7+<br/>
Scala 2.10 or newer.

This will produce jar file in "node2vec_spark/target/"

### Examples
This library has two functions: *randomwalk* and *embedding*. <br/> 
These were described in these papers [node2vec: Scalable Feature Learning for Networks](http://arxiv.org/abs/1607.00653) and [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).

### Random walk
Example:
	
	./spark-submit --class com.navercorp.Main \ 
				   ./node2vec_spark/target/node2vec-0.0.1-SNAPSHOT.jar \
				   --cmd randomwalk --p 100.0 --q 100.0 --walkLength 40 \
				   --input <input> --output <output>

#### Options
Invoke a command without arguments to list available arguments and their default values:

```
--cmd COMMAND
	Functions: randomwalk or embedding. If you want to execute all functions "randomwalk" and "embedding" sequentially input "node2vec". Default "node2vec"
--input [INPUT]
	Input edgelist path. The supported input format is an edgelist: "node1_id_int node2_id_int <weight_float, optional>"
--output [OUTPUT]
	Random paths path.
--walkLength WALK_LENGTH
	Length of walk per source. Default is 80.
--numWalks NUM_WALKS
	Number of walks per source. Default is 10.
--p P
	Return hyperparaemter. Default is 1.0.
--q Q
	Inout hyperparameter. Default is 1.0.
--weighted Boolean
	Specifying (un)weighted. Default is true.
--directed Boolean
	Specifying (un)directed. Default is false.
--degree UPPER_BOUND_OF_NUMBER_OF_NEIGHBORS
	Specifying upper bound of number of neighbors. Default is 30.
--indexed Boolean
	Specifying whether nodes in edgelist are indexed or not. Default is true.
```

* If "indexed" is set to false, *node2vec_spark* index nodes in input edgelist, example: <br/>
  **unindexed edgelist:**<br/>
  node1 	node2 	1.0<br/>
  node2 	node7 	1.0<br/>
  
  **indexed:**<br/>
  1 	2 	1.0<br/>
  2 	3 	1.0<br/>

  1 	node1<br/>
  2 	node2<br/>
  3 	node7

#### Input
The supported input format is an edgelist:

	node1_id_int 	node2_id_int 	<weight_float, optional>
	or
	node1_str 	node2_str 	<weight_float, optional>, Please set the option "indexed" to false


#### Output
The output file (number of nodes)*numWalks random paths as follows:

	src_node_id_int 	node1_id_int 	node2_id_int 	... 	noden_id_int


### Embedding random paths
Example:
	
	./spark-submit --class com.navercorp.Main \
				   ./node2vec_spark/target/node2vec-0.0.1-SNAPSHOT.jar \
				   --cmd embedding --dim 50 --iter 20 \
				   --input <input> --nodePath <node2id_path> --output <output>

#### Options
Invoke a command without arguments to list available arguments and their default values:

```
--cmd COMMAND
	embedding. If you want to execute sequentially all functions: "randomwalk" and "embedding", input "node2vec". default "node2vec"
--input [INPUT]
	Input random paths. The supported input format is an random paths: "src_node_id_int node1_id_int ... noden_id_int"
--output [OUTPUT]
	word2vec model(.bin) and embeddings(.emb).
--nodePath [NODE\_PATH]
	Input node2index path. The supported input format: "node1_str node1_id_int"
--iter ITERATION
	Number of epochs in SGD. Default 10.
--dim DIMENSION
	Number of dimensions. Default is 128.
--window WINDOW_SIZE
	Context size for optimization. Default is 10.

```

#### Input
The supported input format is an random paths:

	src_node_id_int 	node1_id_int 	... 	noden_id_int

#### Output
The output files are **embeddings and word2vec model.** The embeddings file has the following format: 

	node1_str 	dim1 dim2 ... dimd

where dim1, ... , dimd is the d-dimensional representation learned by word2vec.

the output file *word2vec model* has the spark word2vec model format. please reference to https://spark.apache.org/docs/1.5.2/mllib-feature-extraction.html#word2vec

## References
1. [node2vec: Scalable Feature Learning for Networks](http://arxiv.org/abs/1607.00653)
2. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)