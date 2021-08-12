# node2vec

This repository provides a reference implementation of *node2vec* as described in the paper:<br>
> node2vec: Scalable Feature Learning for Networks.<br>
> Aditya Grover and Jure Leskovec.<br>
> Knowledge Discovery and Data Mining, 2016.<br>
> <Insert paper link>

The *node2vec* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. Please check the [project page](https://snap.stanford.edu/node2vec/) for more details.

### Running in Docker
```Bash
docker build -t node2vec:latest .
```

#### Example
```Bash
docker run --rm -v ${PWD}:/node2vec\
                     -i -t node2vec:latest /bin/bash\
                     -c "cd /node2vec;\
                         python src/main.py --input graph/karate.edgelist\
                                            --output emb/karate.emd \
                                            --dimensions 128\
                                            --p 0.9\
                                            --q 0.1;\
                         exit"
```

### Basic Usage

#### Example
To run *node2vec* on Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/karate.edgelist --output emb/karate.emd``

#### Options
You can check out the other options available to use with *node2vec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>

The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The output file has *n+1* lines for a graph with *n* vertices.
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:

	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *node2vec*.

### Citing
If you find *node2vec* useful for your research, please consider citing the following paper:

	@inproceedings{node2vec-kdd2016,
	author = {Grover, Aditya and Leskovec, Jure},
	 title = {node2vec: Scalable Feature Learning for Networks},
	 booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	 year = {2016}
	}


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <adityag@cs.stanford.edu>.

*Note:* This is only a reference implementation of the *node2vec* algorithm and could benefit from several performance enhancement schemes, some of which are discussed in the paper.
