This is a fork from [aditya-grover/node2vec](https://github.com/aditya-grover/node2vec).


To run the code, install packages in `requirements.txt` file.

Check the original repository for instructions to run node2vec.

Adding to the node2vec algorithm, we included more code of our own to replicate the experiments mentioned in the paper and a number of experiments:

## Clustering

Code is available in `src/kmeans.py`. The initial purpose was to reproduce Figure 3 in node2vec paper. Therefore, the default type of analysis is homophily clustering with Les Miserables character graph. To use the code for other purposes, please read below about changing parameter values.

Three types of analyses are supported: 

1. Homophily (community structure) with node2vec embeddings.
2. Structural equivalance with node2vec embeddings.
3. Structural equivalance with struc2vec embeddings. struc2vec embeddings for Les Miserables and [TerroristRel](https://networkrepository.com/TerroristRel.php) data sets are already available. To obtain embeddings for other data sets in Python 3 consider using, for example, [BioNEV](https://github.com/xiangyue9607/BioNEV).

Change the value of the `SWITCH` variable to define the type of analysis. Available options are `homophily`, `str_eq`, `struc2vec`.

Two types of graph data sets are supported:

1. An arbitrary data set in an external edgelist text file.
2. Les Miserables from `nx.generators.social.les_miserables_graph()` because it contains the names of characters in the novel instead of integer identifiers.

Change the value of the `DATA_NAME` variable to define a data set to be used. The value should be the same as in the folder name and the edgelist file name. For example, use `TerroristRel` to import from `graph/TerroristRel/TerroristRel.edgelist`.

Additional parameters are set in the `args` variable:

* node2vec's parameters `D` (dimensionality of embeddings), `P` (return parameter), `K` (context size) and `L` (walk length).
* `edgelist_delim`, default is comma.

A few other parameters depend on the type of analysis and are set automatically. For example, homophily by default sets the in-out parameter `Q` to 0.5 and the number of clusters `n_clusters` to 6. This can be changed in the `set_other_parameters()` function if needed.

The code outputs images into the `images/` folder.



## Classification
Code is available in `src/multi-classification.ipynb` (Jupyter Notebook)
Sections of the notebook: 
- Replicate the classification experiment in section **4.3 Multi-label classification** 
- Grid search on `p` and `q`
- Scalability test (not used in presentataion)
## Link Prediction
- The files src/dataProcessing.py, src/main.py and src/linkPrediction.py are used for link prediction.
- Given an original edgelist, link prediction is performed as follows, where the data used is karate.edgelist:
- 1. To obtain training graph and testing edges, execute the following command from the project home directory:
```python3 src/dataProcessing.py --input_path graph/karate/karate.edgelist --output_train_path graph/karate/train_edges --output_test_path graph/karate/test_edges --testing_data_ratio 0.2```
- 2. To obtain node embeddings, execute the following command from the project home directory: 
```python3 src/main.py --input graph/karate/train_edges --output emb/karate.emb     ```
- 3.  To obtain predictions, execute the following command from the project home directory
```python3 src/linkPrediction.py --original_edges_path graph/karate/karate.edgelist --node_embeddings_path emb/karate.emb --training_edges_path graph/karate/train_edges --test_edges_path graph/karate/test_edges```
