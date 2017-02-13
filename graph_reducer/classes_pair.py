import numpy as np


class ClassesPair:
    bip_adj_mat = np.empty((0, 0))
    """The bipartite adjacency matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s"""
    r = s = -1
    """The classes composing the bipartite graph"""
    n = 0
    """the cardinality of a class"""
    index_map = np.empty((0, 0))
    """A mapping from the bipartite adjacency matrix nodes to the adjacency matrix ones"""
    bip_avg_deg = 0
    """the average degree of the graph"""
    bip_density = 0
    """the average density of the graph"""
    epsilon = 0.0
    """the epsilon parameter"""

    def __init__(self, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_adj_mat.shape[0]
        self.bip_avg_deg = self.bip_avg_degree()
        self.bip_density = self.compute_bip_density()
        self.epsilon = epsilon

    def bip_avg_degree(self):
        """
        compute the average degree of the bipartite graph
        :return the average degree
        """
        return (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        """
        compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the
        bipartite graph
        :return the density
        """
        return float(self.bip_adj_mat.sum()) / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        """
        compute the degree of all vertices in the bipartite graph
        :return a (n,) numpy array containing the degree of each vertex
        """
        c_v_degs = np.sum(self.bip_adj_mat, 0)
        c_v_degs = np.vstack((c_v_degs, np.sum(self.bip_adj_mat, 1)))
        return c_v_degs

    # def neighbourhood_matrix(self, transpose_first=True):
    #     if transpose_first:
    #         return self.bip_adj_mat.T @ self.bip_adj_mat
    #     else:
    #         return self.bip_adj_mat @ self.bip_adj_mat.T
    #
    # def neighbourhood_deviation_matrix(self, nh_mat):
    #     return nh_mat - ((self.bip_avg_deg ** 2.0) / self.n)

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat)
        mat -= (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()

        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            # print "sigma_y = " + str(sigma_y)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        return np.array([])

    def find_Yp(self, degrees, Y_indices):
        # print "min el = " + str(np.min(degrees - self.bip_avg_degree))
        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        sums = np.full((self.n,), -np.inf)
        for i in Yp_indices:
            sums[i] = 0
            for j in list(set(Y_indices) - set(Yp_indices)):
                sums[i] += nh_dev_mat[i, j]
        return np.argmax(sums)

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        cert = list(self.index_map[1][outliers_in_Yp])
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl


class WeightedClassesPair:
    bip_sim_mat = np.empty((0, 0))
    """The bipartite similarity matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s."""
    bip_adj_mat = np.empty((0, 0))
    """The bipartite adjacency matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s"""
    r = s = -1
    """The classes composing the bipartite graph"""
    n = 0
    """the cardinality of a class"""
    index_map = np.empty((0, 0))
    """A mapping from the bipartite adjacency matrix nodes to the adjacency matrix ones"""
    bip_avg_deg = 0.0
    """the average degree of the graph"""
    bip_density = 0.0
    """the average density of the graph"""
    epsilon = 0.0
    """the epsilon parameter"""

    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_sim_mat = sim_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_sim_mat.shape[0]
        self.bip_avg_deg = self.bip_avg_degree()
        self.bip_density = self.compute_bip_density()
        self.epsilon = epsilon

    def bip_avg_degree(self):
        """
        compute the average degree of the bipartite graph
        :return the average degree
        """
        return (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        """
        compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the
        bipartite graph
        :return the density
        """
        return self.bip_sim_mat.sum() / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        """
        compute the degree of all vertices in the bipartite graph
        :return a (n,) numpy array containing the degree of each vertex
        """
        c_v_degs = np.sum(self.bip_adj_mat, 0)
        c_v_degs = np.vstack((c_v_degs, np.sum(self.bip_adj_mat, 1)))
        return c_v_degs

    # def neighbourhood_matrix(self, transpose_first=True):
    #     if transpose_first:
    #         return self.bip_adj_mat.T @ self.bip_adj_mat
    #     else:
    #         return self.bip_adj_mat @ self.bip_adj_mat.T
    #
    # def neighbourhood_deviation_matrix(self, nh_mat):
    #     return nh_mat - ((self.bip_avg_deg ** 2.0) / self.n)

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat)
        mat -= (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()

        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        return np.array([])

    def find_Yp(self, degrees, Y_indices):
        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        sums = np.full((self.n,), -np.inf)
        for i in Yp_indices:
            sums[i] = 0
            for j in list(set(Y_indices) - set(Yp_indices)):
                sums[i] += nh_dev_mat[i, j]
        return np.argmax(sums)

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        cert = list(self.index_map[1][outliers_in_Yp])
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0.0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl