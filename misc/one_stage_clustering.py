import os
import time
import numpy as np
import skimage.transform
from sklearn.cluster import spectral_clustering
import BerkeleyImgSegmentationPerfEvaluator.RI_VOI_computer

from misc import utils
from misc import dominant_sets


root = ".."
main_dir = root + "//test_data"
img_dir_path = root + "//test_data//original"
res_dir_path = root + "//test_data//results"
gt_dir_path = root + "//test_data//groundTruth"
clustering_alg = "DS_n"
n_clusters = np.empty((0, 1))
resize_factor = 0.25

if __name__ == "__main__":
    is_colored = True
    is_weighted = True

    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += "//resize_factor_" + str.replace(str(resize_factor), ".", "_")
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += ("//" + clustering_alg)
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    res_dir_path += "//one_stage"
    if not os.path.exists(os.path.normpath(res_dir_path)):
        os.mkdir(os.path.normpath(res_dir_path))

    log = open(os.path.normpath(res_dir_path + "//FK_Sz1.log"), "w")
    log.truncate()
    data = open(os.path.normpath(res_dir_path + "//" + clustering_alg + "1.csv"),"w")
    data.truncate()
    data.write("cf,t,PRI,VOI" + "\n")
    RI_VOI_comp = BerkeleyImgSegmentationPerfEvaluator.RI_VOI_computer.RIVOIComputer(1)

    for img_fname in sorted(os.listdir(os.path.normpath(img_dir_path))):
        print("processing image " + img_fname)

        img_pname = os.path.normpath(img_dir_path + "//" + img_fname)
        img_name = os.path.splitext(img_fname)[0]
        curr_res_dir_path = os.path.normpath(res_dir_path + "//" + img_name)
        if not os.path.exists(os.path.normpath(curr_res_dir_path)):
            os.mkdir(os.path.normpath(curr_res_dir_path))

        graph_mat, original_img_shape, resized_img_shape = utils.from_image_to_adj_mat(img_pname, 1.0, is_colored,
                                                                                       is_weighted, 1.0, resize_factor)

        log.write("image " + img_fname + ". density = " + str(utils.graph_density(graph_mat)) + "\n")

        start_time = time.time()
        if clustering_alg == "SC_n":
            results = spectral_clustering(graph_mat, 4) + 1
        elif clustering_alg == "DS_n":
            results = dominant_sets.dominant_sets(graph_mat) + 1
        else:
            raise ValueError("Incorrect clustering algorithm")

        del graph_mat
        results = results.reshape(resized_img_shape)
        elapsed_time = time.time() - start_time

        clusters_found = int(np.max(results))
        n_clusters = np.vstack((n_clusters, np.array([clusters_found])))

        results = skimage.transform.resize(results, original_img_shape, 0, preserve_range=True).astype(int)
        RI_VOI_comp.set_seg(results)

        gts, _ = utils.read_gts(gt_dir_path, img_name + ".mat")

        for gt in gts:
            RI_VOI_comp.set_gt(gt)
            RI_VOI_comp.compute_RI_and_VOI_sums()
        RI_VOI_comp.update_partial_values(0, len(gts))

        print("finished. saving data")
        log.write("  clusters found = " + str(int(clusters_found)).zfill(2) + ". time = " + str(elapsed_time) + "\n")
        data.write(str(int(clusters_found)).zfill(2) + "," + str(elapsed_time) + "," + str(RI_VOI_comp.avgRI) + "," + str(RI_VOI_comp.avgVOI) + "\n")
        np.save(os.path.normpath(curr_res_dir_path + "//" + img_name + "_cl_" + str(int(clusters_found)) + ".npy"), results)
    np.save(os.path.normpath(res_dir_path + "//n_clusters.npy"), n_clusters)
