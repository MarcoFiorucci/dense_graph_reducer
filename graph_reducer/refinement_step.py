import numpy as np
import random
import sys


def randoramized(self):
    """
    perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes in a random way. Some heuristic is applied in order to
    speed up the process.
    """
    pass

def get_s_r_degrees(self, s, r):
    """
    Computes the degrees of s and r with respect to each other.
    s: index of nodes of class s
    r: index of nodes of class r
    return: a numpy array where each index correspondent to a node (in s or r) contains the calculated degree, zero otherwise.
    """

    # Gets the indices of elements which are part of class s (same for r)
    s_indices = np.where(self.classes == s)[0]
    r_indices = np.where(self.classes == r)[0]

    # Isolate the columns the adj_mat of the nodes of class s (same for r)
    s_columns = self.adj_mat[:, s_indices]
    r_columns = self.adj_mat[:, r_indices]

    # Get the degrees of s w.r.t. the elements of r (same for r)
    s_degs = r_columns[s_indices, :].sum(1)
    r_degs = s_columns[r_indices, :].sum(1)

    s_r_degs = np.zeros(len(self.degrees))

    # Put the degree of s the indices of s (same for r)
    s_r_degs[s_indices] = s_degs
    s_r_degs[r_indices] = r_degs

    return s_r_degs.astype(int)


def degree_based(self):
    """
    perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes according to their degree. Some heuristic is applied in order to
    speed up the process
    """

    to_be_refined = list(range(1, self.k + 1))
    irregular_r_indices = []
    is_classes_cardinality_odd = self.classes_cardinality % 2 == 1
    self.classes_cardinality //= 2

    while to_be_refined:
        s = to_be_refined.pop(0)

        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        if irregular_r_indices:
            np.random.seed(314)
            random.seed(314)
            chosen = random.choice(irregular_r_indices)
            to_be_refined.remove(chosen)
            irregular_r_indices = []

            s_r_degs = get_s_r_degrees(self, s, chosen)

            # i = 0 for r, i = 1 for s
            for i in [0, 1]:
                cert_length = len(self.certs_compls_list[chosen - 2][s - 1][0][i])
                compl_length = len(self.certs_compls_list[chosen - 2][s - 1][1][i])

                greater_set_ind = np.argmax([cert_length, compl_length])
                lesser_set_ind = np.argmin(
                    [cert_length, compl_length]) if cert_length != compl_length else 1 - greater_set_ind

                greater_set = self.certs_compls_list[chosen - 2][s - 1][greater_set_ind][i]
                lesser_set = self.certs_compls_list[chosen - 2][s - 1][lesser_set_ind][i]

                self.classes[lesser_set] = 0

                difference = len(greater_set) - self.classes_cardinality
                # retrieve the first <difference> nodes sorted by degree.
                # N.B. NODES ARE SORTED IN DESCENDING ORDER
                difference_nodes_ordered_by_degree = sorted(greater_set, key=lambda el: s_r_degs[el], reverse=True)[0:difference]

                self.classes[difference_nodes_ordered_by_degree] = 0
        else:
            self.k += 1

            s_indices_ordered_by_degree = sorted(list(np.where(self.classes == s)[0]), key=lambda el: s_r_degs[el], reverse=True)

            if is_classes_cardinality_odd:
                self.classes[s_indices_ordered_by_degree.pop(0)] = 0
            self.classes[s_indices_ordered_by_degree[0:self.classes_cardinality]] = self.k

    C0_cardinality = np.sum(self.classes == 0)
    num_of_new_classes = C0_cardinality // self.classes_cardinality
    nodes_in_C0_ordered_by_degree = np.array([x for x in self.degrees if x in np.where(self.classes == 0)[0]])
    for i in range(num_of_new_classes):
        self.k += 1
        self.classes[nodes_in_C0_ordered_by_degree[
                     (i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = self.k

    C0_cardinality = np.sum(self.classes == 0)
    if C0_cardinality > self.epsilon * self.N:
        sys.exit("Error: not enough nodes in C0 to create a new class."
                 "Try to increase epsilon or decrease the number of nodes in the graph")
