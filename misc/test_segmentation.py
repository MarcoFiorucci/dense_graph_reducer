import argparse
import os
import time
import skimage.transform

import numpy as np
from misc import utils
from misc import dominant_sets
from matplotlib import pyplot
from sklearn.cluster import spectral_clustering

from graph_reducer.szemeredi_lemma_builder import generate_szemeredi_reg_lemma_implementation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="The source images path")
    parser.add_argument("resize_factor", help="Resize factor of the image (from 0.0 to 1.0)", type=float)
    parser.add_argument("noise_factor", help="Noise percentage in the image (from 0.0 to 1.0)", type=float)
    parser.add_argument("alg_kind",
                        help="The kind of algorithm to perform graph summarization. Currently 'alon' and"
                             " 'frieze_kannan' are supported")
    parser.add_argument("epsilon", help="The epsilon parameter ", type=float)
    parser.add_argument("b", help="The cardinality of the initial partition", type=int)
    parser.add_argument("compression_rate",
                        help="The minimum compression rate (from 0.0 to 1.0. Small values, high compression)",
                        type=float)
    parser.add_argument("-w", "--is_weighted", help="Whether the graph is weighted or not", action="store_true")
    parser.add_argument("-c", "--is_colored", help="Whether the image is colored or not", action="store_true")
    parser.add_argument("-r", "--reconstruct", help="Reconstruct the original graph from the reduced one",
                        action="store_true")
    parser.add_argument("--random_initialization", help="Is the initial partition random, or degree based?",
                        action="store_true")
    parser.add_argument("--random_refinement",
                        help="Is the refinement step random, or degree based? (random refinement not implemented yet)",
                        action="store_true")
    parser.add_argument("--drop_edges_between_irregular_pairs",
                        help="Consider the reduced matrix to be fully connected or drop edges between irregular pairs",
                        action="store_true")
    parser.add_argument("--iteration_by_iteration",
                        help="perform the algorithm iteration by iteration (useful if debug is set to True)",
                        action="store_true")
    parser.add_argument("-d", "--debug", help="print useful debug info", action="store_true")
    parser.add_argument("--src_plot", help="Plot the source image segmentation", action="store_true")
    parser.add_argument("--dst_plot", help="Plot the compressed image segmentation", action="store_true")
    parser.add_argument("--reconstr_plot",
                        help="Plot the reconstructed image segmentation (ignored if --reconstruct option is set to"
                             " false", action="store_true")
    args = parser.parse_args()

    for img_path in sorted(os.listdir(os.path.normpath(args.image_path))):
        print("processing image " + img_path)

        img_path = os.path.normpath(args.image_path + "//" + img_path)

        graph_mat, original_img_shape, resized_img_shape = utils.from_image_to_adj_mat(os.path.normpath(img_path), 1.0,
                                                                                       args.is_colored,
                                                                                       args.is_weighted,
                                                                                       1.0, args.resize_factor,
                                                                                       args.noise_factor)
        print("graph_mat density = " + str(utils.graph_density(graph_mat)))
        print()

        start_time_one_stage_clustering = time.time()

        labels = dominant_sets.dominant_sets(graph_mat)
        # labels = spectral_clustering(graph_mat, 2)
        labels = np.asarray(labels).reshape(resized_img_shape)
        one_stage_clustering_time = time.time() - start_time_one_stage_clustering

        labels = skimage.transform.resize(labels, original_img_shape, 0, preserve_range=True).astype(int)

        if args.src_plot:
            pyplot.matshow(labels)
            pyplot.show()

        start_time_two_stage_clustering = time.time()

        alg = generate_szemeredi_reg_lemma_implementation(args.alg_kind, graph_mat, args.epsilon, args.is_weighted,
                                                          args.random_initialization, args.random_refinement,
                                                          args.drop_edges_between_irregular_pairs)
        alg.run(args.b, args.compression_rate, args.iteration_by_iteration, args.debug)

        print()
        print()

        reduced_labels = dominant_sets.dominant_sets(alg.reduced_sim_mat)
        # reduced_labels = spectral_clustering(alg.reduced_sim_mat, 2)
        results = np.zeros(alg.classes.shape)

        for i in range(1, alg.k + 1):
            results[alg.classes == i] = reduced_labels[i - 1]

        results = results.reshape(resized_img_shape)
        two_stage_clustering_time = time.time() - start_time_two_stage_clustering

        results = skimage.transform.resize(results, original_img_shape, 0, preserve_range=True).astype(int)

        if args.dst_plot:
            pyplot.matshow(results)
            pyplot.show()

        print('Computational Time One-Stage Clustering = ' + str(one_stage_clustering_time))
        print('Computational Time Two-Stage Clustering = ' + str(two_stage_clustering_time))
        print()

        if args.reconstruct:
            reconstruct_mat = alg.reconstruct_original_mat(0.5)
            labels_reconstr_mat = spectral_clustering(reconstruct_mat, 2)
            labels_reconstr_mat = np.asarray(labels_reconstr_mat).reshape(resized_img_shape)
            labels_reconstr_mat = skimage.transform.resize(labels_reconstr_mat, original_img_shape, 0,
                                                           preserve_range=True).astype(int)

            if args.reconstr_plot:
                pyplot.matshow(labels_reconstr_mat)
                pyplot.show()
