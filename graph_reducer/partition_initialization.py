import numpy as np


def random(self, b=2):
    """
    perform step 1 of Alon algorithm,  initializing the starting partition of a graph G.
    The result is stored in "classes" attribute
    :param  G: the similarity matrix or adjacency matrix of G
    :param  b: the number of classes in which the vertices are partitioned
    :param  random: if set to True, the vertices are assigned randomly to the classes,
            if set to False, node are decreasingly ordered by their degree and then splitted
            in classes
    """
    self.k = b
    self.classes = np.zeros(self.N)
    self.classes_cardinality = self.N // self.k

    for i in range(self.k):
        self.classes[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)] = i + 1

    np.random.shuffle(self.classes)


def degree_based(self, b=2):
    """
    perform step 1 of Alon algorithm,  initializing the starting partition of a graph G.
    The result is stored in "classes" attribute
    :param  G: the similarity matrix or adjacency matrix of G
    :param  b: the number of classes in which the vertices are partitioned
    :param  random: if set to True, the vertices are assigned randomly to the classes,
            if set to False, node are decreasingly ordered by their degree and then splitted
            in classes
    """
    self.k = b
    self.classes = np.zeros(self.N)
    self.classes_cardinality = self.N // self.k

    for i in range(self.k):
        self.classes[self.degrees[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = i + 1
