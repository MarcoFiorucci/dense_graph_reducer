import os
import time
import numpy as np
import skimage.transform
import scipy as sc
from scipy import misc
from scipy import spatial
from scipy import stats
from scipy import io


def from_image_to_adj_mat(img_path, sigma=2.0, is_colored=False, is_weighted=False, thresh_factor=1.0,
                          resize_factor=1.0, debug=False):
    """
    This function convert a standard image to an adjacency matrix
    :param img_path: path of the image
    :param sigma: the standard deviation
    :param is_colored: read the image as colored or grayscale
    :param is_weighted: determines if the resulting matrix should be a similarity or adjacency one
    :param thresh_factor: a scale factor to threshold the similarity matrix to obtain an adjacency one
    :param resize_factor: parameter to optionally resize the image before generating the graph matrix
    :param debug: if set to True additional debug info is printed
    :return:
    """
    if debug:
        start_time = time.time()

    # Read an image
    if is_colored:
        img = misc.imread(img_path, mode='RGB')
    else:
        img = misc.imread(img_path, mode='L')

    original_img_shape = img.shape[0:2]

    # resize the image if necessary
    if resize_factor != 1.0:
        img = skimage.transform.resize(img, (
        np.ceil(resize_factor * original_img_shape[0]), np.ceil(resize_factor * original_img_shape[1])))

    resized_img_shape = img.shape[0:2]

    # reshape and normalize vectors
    if is_colored:
        img = img.reshape(img.shape[0] * img.shape[1], 3)
    else:
        img = img.reshape(img.shape[0] * img.shape[1], 1)
    img = img / np.max(img, 0)

    # Compute the distance matrix
    T = sc.spatial.distance.cdist(img, img, 'euclidean')

    # Compute the similarity matrix using a Gaussian kernel: T = np.exp(-(T ** 2.0 / sigma ** 2.0))
    T **= 2.0
    T /= sigma**2.0
    T *= -1.0
    np.exp(T, T)

    # Set the main diagonal to zero
    np.fill_diagonal(T, 0.0)

    avg_sim = T.sum() / T.size
    threshold = thresh_factor * avg_sim
    if is_weighted:
        T = sc.stats.threshold(T, threshold, np.inf, 0.0)
    else:
        # Compute the adjacency matrix
        T = sc.stats.threshold(T, -np.inf, threshold, 1.0)
        T = sc.stats.threshold(T, threshold, np.inf, 0.0)

    if debug:
        time_elapsed = (time.time() - start_time)
        print("Computational time (From Image to Adjacency Matrix)")
        print(time_elapsed)

    return T, original_img_shape, resized_img_shape


def threshold_sim_mat(sim_mat, threshold):
    """
    generate the adjacency matrix, thresholding the elements of a similarity matrix
    :param sim_mat: the similarity matrix to be thresholded
    :param thresh_percentage: the threshold percentage, this value is then multiplied by the average weight of the
           graph
    :return the adjacency matrix
    """
    return (sim_mat >= threshold).astype(int)


def graph_density(graph_mat):
    """
    compute the density of undirected graphs
    :param graph_mat: the similarity/adjacency matrix representing the graph
    :return the density of the graph
    """
    return float(graph_mat.sum()) / float(graph_mat.shape[0] * (graph_mat.shape[0] - 1))


def generate_szemeredi_segs(iid, res_dir_path):
    segs = []
    res_dir_path = os.path.normpath(res_dir_path)

    img_name = sorted(os.listdir(res_dir_path))[iid]
    for seg_name in sorted(os.listdir(os.path.normpath(res_dir_path + "//" + img_name))):
        segs.append(np.load(res_dir_path + "//" + img_name + "//" + seg_name))
    return segs


def read_gts(mat_dir_path, mat_fname):
    mat = sc.io.loadmat(os.path.normpath(mat_dir_path + "//" + mat_fname))['groundTruth']
    img_shape = mat[0][0]['Segmentation'][0][0].shape
    gts = np.empty((mat.shape[1], img_shape[0], img_shape[1]))
    total_num_of_gts_regs = 0
    for i in range(mat.shape[1]):
        gts[i] = mat[0][i]['Segmentation'][0][0]
        total_num_of_gts_regs += int(np.max(gts[i]))
    return gts, total_num_of_gts_regs
