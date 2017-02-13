import numpy as np
import scipy.sparse.linalg
import math
import sys


def alon1(self, cl_pair):
    """
    verify the first condition of Alon algorithm (regularity of pair)
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    :return: A list of two empty lists representing the empty certificates
    :return: A list of two empty lists representing the empty complements
    """
    return cl_pair.bip_avg_deg < (self.epsilon ** 3.0) * cl_pair.n, [[], []], [[], []]


def alon2(self, cl_pair):
    """
    verify the second condition of Alon algorithm (irregularity of pair)
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    certs = []
    compls = []
    s_vertices_degrees = cl_pair.classes_vertices_degrees()[1, :]
    deviated_nodes = np.abs(s_vertices_degrees - cl_pair.bip_avg_deg) > (self.epsilon ** 4.0) * cl_pair.n
    deviation_threshold = (self.epsilon ** 4.0) * cl_pair.n
    # retrieve positive deviated nodes
    one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg > deviation_threshold)
    is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n

    if is_irregular:
        certs.append(list(cl_pair.index_map[0][range(cl_pair.n)]))
        certs.append(list(cl_pair.index_map[1][one_direction_nodes]))
        compls.append([])
        compls.append(list(cl_pair.index_map[1][~one_direction_nodes]))
    else:
        # retrieve negative deviated nodes
        one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg < -deviation_threshold)
        is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n
        if is_irregular:
            certs.append(list(cl_pair.index_map[0][range(cl_pair.n)]))
            certs.append(list(cl_pair.index_map[1][one_direction_nodes]))
            compls.append([])
            compls.append(list(cl_pair.index_map[1][~one_direction_nodes]))
        else:
            certs = [[], []]
            compls = [[], []]

    return is_irregular, certs, compls


def alon3(self, cl_pair, fast_convergence=True):
    """
    verify the third condition of Alon algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :param fast_convergence: apply the fast convergence version of condition 3
    :return: True if the condition is verified, False otherwise
    """

    is_irregular = False

    cert_s = []
    compl_s = []
    y0 = -1

    # nh_mat = cl_pair.neighbourhood_matrix()
    # nh_dev_mat = cl_pair.neighbourhood_deviation_matrix(nh_mat)
    # s_degrees = np.diag(nh_mat)
    nh_dev_mat, s_degrees = cl_pair.neighbourhood_deviation_matrix()
    if fast_convergence:
        Y_indices = cl_pair.find_Y(nh_dev_mat)

        if not list(Y_indices):
            # enter in Y spurious condition
            is_irregular = True
            return is_irregular, [[], []], [[], []]

        Y_degrees = s_degrees[Y_indices]
        Yp_indices = cl_pair.find_Yp(Y_degrees, Y_indices)

        if not list(Yp_indices):
            # enter in Yp spurious condition
            is_irregular = False
            return is_irregular, [[], []], [[], []]

        y0 = cl_pair.compute_y0(nh_dev_mat, Y_indices, Yp_indices)

        cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, Yp_indices)
    else:
        s_indices = cl_pair.find_Yp(s_degrees, np.arange(cl_pair.n))

        for y0 in s_indices:
            cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, s_indices)
            if cert_s:
                break
    cert_r, compl_r = cl_pair.find_r_cert_and_compl(y0)

    if cert_r and cert_s:
        is_irregular = True
    else:
        cert_r = []
        cert_s = []
        compl_r = []
        compl_s = []

    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]


def frieze_kannan(self, cl_pair):
    """
    verify the condition of Frieze and Kannan algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    cert_r = []
    cert_s = []
    compl_r = []
    compl_s = []

    if self.is_weighted:
        W = cl_pair.bip_sim_mat - cl_pair.bip_density
    else:
        W = cl_pair.bip_adj_mat - cl_pair.bip_density

    x, sv_1, y = scipy.sparse.linalg.svds(W, k=1, which='LM')

    is_irregular = (sv_1 >= self.epsilon * cl_pair.n)

    if is_irregular:
        beta = 3.0 / self.epsilon
        x = x.ravel()
        y = y.ravel()
        hat_thresh = beta / math.sqrt(cl_pair.n)
        x_hat = np.where(np.abs(x) <= hat_thresh, x, 0.0)
        y_hat = np.where(np.abs(y) <= hat_thresh, y, 0.0)

        quadratic_threshold = (self.epsilon - 2.0 / beta) * (cl_pair.n / 4.0)

        x_mask = x_hat > 0
        y_mask = y_hat > 0
        x_plus = np.where(x_mask, x_hat, 0.0)
        x_minus = np.where(~x_mask, x_hat, 0.0)
        y_plus = np.where(y_mask, y_hat, 0.0)
        y_minus = np.where(~y_mask, y_hat, 0.0)

        r_mask = np.empty((0, 0))
        s_mask = np.empty((0, 0))

        q_plus = y_plus * 1.0 / hat_thresh
        q_minus = y_minus * 1.0 / hat_thresh

        if x_plus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) >= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_plus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) >= 0.0
            s_mask = (r_mask @ W) <= 0.0
        elif x_minus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) <= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_minus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) <= 0.0
            s_mask = (r_mask @ W) <= 0.0
        else:
            sys.exit("no condition on the quadratic form was verified")

        cert_r = list(cl_pair.index_map[0][r_mask])
        compl_r = list(cl_pair.index_map[0][~r_mask])
        cert_s = list(cl_pair.index_map[1][s_mask])
        compl_s = list(cl_pair.index_map[1][~s_mask])
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]
