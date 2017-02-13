import numpy as np
import random
import sys


def randomized(self):
    """
    perform step 4 of Alon algorithm, performing the refinement of the pairs, processing nodes in a random way. Some heuristic is applied in order to
    speed up the process.
    """
    pass


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
            chosen = random.choice(irregular_r_indices)
            to_be_refined.remove(chosen)
            irregular_r_indices = []

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
                difference_nodes_ordered_by_degree = sorted(greater_set,
                                                            key=lambda el: np.where(self.degrees == el)[0],
                                                            reverse=True)[0:difference]

                self.classes[difference_nodes_ordered_by_degree] = 0
        else:
            self.k += 1
            s_indices_ordered_by_degree = sorted(list(np.where(self.classes == s)[0]),
                                                 key=lambda el: np.where(self.degrees == el)[0], reverse=True)
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
