# graph_reducer
This repository contains a Python 3.x implementation of some constructive versions of Szemeredi Regularity Lemma
for graph summarization.

If you use this code please cite the following papers:

M. Fiorucci, A. Torcinovich, M. Curado, F. Escolano, M. Pelillo. On the Interplay Between Strong Regularity and Graph Densification. In Lecture Notes in Computer Science, Springer, GbRPR 2017.

Pelillo, M., Elezi, I., & Fiorucci, M. (2016). Revealing structure in large graphs: Szemerédi’s regularity lemma and its use in pattern recognition. Pattern Recognition Letters.

Marco and Alessandro

## Contacts
For any question or to point out bugs, please contact marco.fiorucci@unive.it

## Prerequisites
The code makes heavy use of [NumPy](http://www.numpy.org/). Install it using pip:
```
~ ➤ pip3 install --user numpy
```

## Package Installation
To install the package just type the following commands:
```
~ ➤ cd path/to/folder/graph_reducer
graph_reducer ➤ pip3 install .
```
Consider to use `--upgrade` if the package was already installed. For
Anaconda users, remember to `(source) activate` your Anaconda environment first.

## Example of Usage
The code is provided with a small example (`test_segmentation.py`). To run it you need a folder containing images (like the one provided in our repository containing images from the [Berkeley Image Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)), then run the
the algorithm as in the following:
```
src ➤ ls
classes_pair.py  partition_initialization.py  szemeredi_regularity_lemma.py
conditions.py	 refinement_step.py	      szemeredi_utils.py
main.py		 szemeredi_lemma_builder.py
src ➤ python3 main.py "/path/to/images/folder/" --src_plot --dst_plot
```
For each image in the folder, the image will be converted to an adjacency matrix, then segmentated and the segmentation
will be displayed. After this Alon algorithm will be applied and another segmentation will be performed on the reduced similarity matrix. Then the
segmentation will be converted to the original size and will be displayed.

## Parameters for the Example of Usage
The parameters of `test_segmentation.py` are:
- **image_path**: the source images path
- **resize_factor**: resize factor of the image (from 0.0 to 1.0)
- **noise_factor**: noise percentage in the image (from 0.0 to 1.0)
- **alg_kind**: the kind of algorithm to perform graph summarization. Currently 'alon' and 'frieze_kannan' are supported 
- **epsilon**: the epsilon parameter
- **b**: the cardinality of the initial partition
- **compression_rate**: the compression rate (from 0.0 to 1.0. Small values, high compression)
- **-w**, **--is_weighted**: wether the graph is weighted or not
- **-c**, **--is_colored**: whether the image is colored or not
- **--reconstruct**: reconstruct the original graph from the reduced one
- **--random_initialization**: is the initial partition random, or degree based?
- **--random_refinement**: is the refinement step random, or degree based? (random refinement not implemented yet)
- **--drop_edges_between_irregular_pairs**: consider the reduced matrix to be fully connected or drop edges between irregular pairs
- **--iteration_by_iteration**: perform the algorithm iteration by iteration (useful if debug is set to True)
- **-d**, **--debug**: print useful debug info
- **--src_plot**: plot the source image segmentation
- **--dst_plot**: plot the compressed image segmentation
- **--reconstr_plot**: plot the reconstructed image segmentation (ignored if --reconstruct option is set to false

## MATLAB® Wrappers
The code contains also two MATLAB® functions:
- **img2simmat.m**: it takes in input an image path and returns its
adjacency matrix representation
- **summarizegraph.m**: performs a graph summarization method algorithm and takes in input an
 adjacency matrix compressing it and returning its the reduced version
