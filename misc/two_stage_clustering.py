import os
import time
import numpy as np
import skimage.transform
from sklearn.cluster import spectral_clustering
from matplotlib import pyplot
import BerkeleyImgSegmentationPerfEvaluator.RI_VOI_computer

from misc import dominant_sets
from misc import utils
from graph_reducer.szemeredi_lemma_builder import generate_szemeredi_reg_lemma_implementation


root = "..//batches"
main_dir = root + "//batch19"
img_dir_path = main_dir + "//original"
gt_dir_path = main_dir + "//groundTruth"
res_dir_path = main_dir + "//results"

clustering_alg = "DS_n"
alg_kind = 'frieze_kannan'
resize_factor = 0.25
n_clusters = np.empty((0, 3))
Al_epsilons = np.arange(0.3, 0.75, 0.05)
FK_epsilons = np.arange(0.2, 0.42, 0.02)

if __name__ == "__main__":
    is_colored = False
    is_weighted = True
    random_initialization = False
    random_refinement = False
    drop_edges_between_irregular_pairs = False
    reconstruct = False

    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += "//resize_factor_" + str.replace(str(resize_factor), ".", "_")
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += ("//" + clustering_alg)
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += "//two_stage"
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += "//" + alg_kind
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    data = open(os.path.normpath(res_dir_path + "//" + clustering_alg + "2.csv"), "w")
    data.truncate()
    log = open(os.path.normpath(res_dir_path + "//" + clustering_alg + "_FK_Sz2.log"), 'w')
    log.truncate()

    if reconstruct:
        data.write("iid,eps,k,cc,cf,t_compr,t_clust,t_recon,PRI_red,VOI_red,PRI_rec,VOI_rec\n")
    else:
        data.write("iid,eps,k,cc,cf,t_compr,t_clust,PRI_red,VOI_red\n")

    RI_VOI_comp = BerkeleyImgSegmentationPerfEvaluator.RI_VOI_computer.RIVOIComputer(1)
    RI_VOI_rec_comp = BerkeleyImgSegmentationPerfEvaluator.RI_VOI_computer.RIVOIComputer(1)

    for iid, img_fname in enumerate(sorted(os.listdir(os.path.normpath(img_dir_path)))):
        print("Image " + img_fname)
        log.write("image " + img_fname + "\n")

        img_pname = os.path.normpath(img_dir_path + "//" + img_fname)
        img_name = os.path.splitext(img_fname)[0]
        curr_res_dir_path = os.path.normpath(res_dir_path + "//" + img_name)

        if not os.path.exists(os.path.normpath(curr_res_dir_path)):
            os.mkdir(os.path.normpath(curr_res_dir_path))

        if clustering_alg == 'DS_n':
            if alg_kind == 'alon':
                epsilons = Al_epsilons
            elif alg_kind == 'frieze_kannan':
                epsilons = FK_epsilons
            else:
                raise ValueError("Could not find the specified graph summarization method")
        elif clustering_alg == 'SC_n':
            n_clusters = np.load(os.path.normpath(res_dir_path + "//n_clusters.npy"))
            epsilons = n_clusters[np.where(n_clusters[:, 1] == iid)[0], 1]
            clusters_found = n_clusters[np.where(n_clusters[:, 1] == iid)[0], 2]
            ind = 0

        graph_mat, original_img_shape, resized_img_shape = utils.from_image_to_adj_mat(img_pname, 1.0, is_colored,
                                                                                       is_weighted, 1.0, resize_factor)
        for eps in epsilons:
            print("    epsilon = " + str(eps))

            start_time = time.time()
            alg = generate_szemeredi_reg_lemma_implementation(alg_kind, graph_mat, eps, is_weighted,
                                                              random_initialization, random_refinement,
                                                              drop_edges_between_irregular_pairs)
            alg.run(2, compression_rate=256, iteration_by_iteration=False, verbose=False)

            if 4 <= alg.k <= 256:
                compression_time = time.time() - start_time
                if clustering_alg == 'SC_n':
                    reduced_labels = spectral_clustering(alg.reduced_sim_mat, clusters_found[ind]) + 1
                    ind += 1
                elif clustering_alg == 'DS_n':
                    reduced_labels = dominant_sets.dominant_sets(alg.reduced_sim_mat) + 1
                else:
                    raise ValueError("incorrect clustering algorithm")

                reduced_results = np.ones(alg.classes.shape)
                for i in range(1, alg.k + 1):
                    reduced_results[alg.classes == i] = reduced_labels[i - 1]

                reduced_results = reduced_results.reshape(resized_img_shape)
                clustering_time = time.time() - (start_time + compression_time)

                if reconstruct:
                    reconstruct_mat = alg.reconstruct_original_mat(0.25)
                    reconstruction_time = time.time() - (start_time + compression_time + clustering_time)

                clusters_found = int(np.max(reduced_results))
                n_clusters = np.vstack((n_clusters, np.array([iid, eps, clusters_found])))
                reduced_results = skimage.transform.resize(reduced_results, original_img_shape, 0,
                                                           preserve_range=True).astype(int)

                if reconstruct:
                    if clustering_alg == 'SC_n':
                        labels_reconstr_mat = spectral_clustering(reconstruct_mat, 2) + 1
                    elif clustering_alg == 'DS_n':
                        labels_reconstr_mat = dominant_sets.dominant_sets(reconstruct_mat) + 1
                    else:
                        raise ValueError("incorrect clustering algorithm")

                    labels_reconstr_mat = np.asarray(labels_reconstr_mat).reshape(resized_img_shape)
                    labels_reconstr_mat = skimage.transform.resize(labels_reconstr_mat, original_img_shape, 0,
                                                                   preserve_range=True).astype(int)

                RI_VOI_comp.set_seg(reduced_results)
                if reconstruct:
                    RI_VOI_rec_comp.set_seg(labels_reconstr_mat)
                gts, _ = utils.read_gts(gt_dir_path, img_name + ".mat")

                for gt in gts:
                    RI_VOI_comp.set_gt(gt)
                    RI_VOI_comp.compute_RI_and_VOI_sums()
                    if reconstruct:
                        RI_VOI_rec_comp.set_gt(gt)
                        RI_VOI_rec_comp.compute_RI_and_VOI_sums()

                RI_VOI_comp.update_partial_values(0, len(gts))
                if reconstruct:
                    RI_VOI_rec_comp.update_partial_values(0, len(gts))

                if reconstruct:
                    log.write("  eps = " + str(format(eps, '.2f')) +
                              ". k = " + str(int(alg.k)).zfill(3) +
                              ". classes cardinality = " + str(alg.classes_cardinality).zfill(4) +
                              ". clusters found = " + str(int(clusters_found)).zfill(2) +
                              ". compression time = " + format(compression_time, '.4f') +
                              ". clustering time = " + format(clustering_time, '.4f') +
                              ". reconstruction time = " + format(reconstruction_time, '.4f') + "\n")
                else:
                    log.write("  eps = " + str(format(eps, '.2f')) +
                              ". k = " + str(int(alg.k)).zfill(3) +
                              ". classes cardinality = " + str(alg.classes_cardinality).zfill(4) +
                              ". clusters found = " + str(int(clusters_found)).zfill(2) +
                              ". compression time = " + format(compression_time, '.4f') +
                              ". clustering time = " + format(clustering_time, '.4f') + "\n")

                if reconstruct:
                    data.write(img_name.zfill(6) + "," +
                               format(eps, '.2f') + "," +
                               str(int(alg.k)).zfill(3) + "," +
                               str(alg.classes_cardinality).zfill(4) + "," +
                               str(clusters_found).zfill(2) + "," +
                               format(compression_time, '.4f').zfill(8) + "," +
                               format(clustering_time, '.4f').zfill(8) + "," +
                               format(reconstruction_time, '.4f').zfill(8) + "," +
                               format(RI_VOI_comp.avgRI, '.4f') + "," +
                               format(RI_VOI_comp.avgVOI, '.4f') + "," +
                               format(RI_VOI_rec_comp.avgRI, '.4f') + "," +
                               format(RI_VOI_rec_comp.avgVOI, '.4f') + "\n")
                else:
                    data.write(img_name.zfill(6) + "," +
                               format(eps, '.2f') + "," +
                               str(int(alg.k)).zfill(3) + "," +
                               str(alg.classes_cardinality).zfill(4) + "," +
                               str(clusters_found).zfill(2) + "," +
                               format(compression_time, '.4f').zfill(8) + "," +
                               format(clustering_time, '.4f').zfill(8) + "," +
                               format(RI_VOI_comp.avgRI, '.4f') + "," +
                               format(RI_VOI_comp.avgVOI, '.4f') + "," + "\n")

                # + "," + str.replace(str.replace(str(alg.index_vec), "[", ""), "]",
                #                     "") + ",0.0" * (8 - len(alg.index_vec))
                np.save(os.path.normpath(curr_res_dir_path + "//" + img_name +
                                         "_eps_" + str.replace(format(eps, '.2f'), ".", "_") +
                                         "_k_" + str(int(alg.k)) +
                                         "_red.npy"),
                        reduced_results)

                if reconstruct:
                    np.save(os.path.normpath(curr_res_dir_path + "//" + img_name +
                                             "_eps_" + str.replace(format(eps, '.2f'), ".", "_") +
                                             "_k_" + str(int(alg.k)) +
                                             "_rec.npy"),
                            labels_reconstr_mat)

                pyplot.imshow(reduced_results)
                pyplot.savefig(filename=os.path.normpath(curr_res_dir_path + "//" + img_name +
                                                         "_eps_" + str.replace(format(eps, '.2f'), ".", "_") +
                                                         "_red.png"))
                if reconstruct:
                    pyplot.imshow(labels_reconstr_mat)
                    pyplot.savefig(filename=os.path.normpath(curr_res_dir_path + "//" + img_name +
                                                             "_eps_" + str.replace(format(eps, '.2f'), ".", "_") +
                                                             "_rec.png"))

            del alg
        del graph_mat
    if clustering_alg == 'DS_n':
        np.save(os.path.normpath(res_dir_path + "//n_clusters.npy"), n_clusters)
