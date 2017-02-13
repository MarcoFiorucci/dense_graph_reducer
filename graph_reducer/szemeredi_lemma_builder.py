import graph_reducer.szemeredi_regularity_lemma as srl
from graph_reducer import partition_initialization
from graph_reducer import refinement_step
from graph_reducer import conditions


def generate_szemeredi_reg_lemma_implementation(kind, sim_mat, epsilon, is_weighted, random_initialization,
                                                random_refinement, drop_edges_between_irregular_pairs):
    """
    generate an implementation of the Szemeredi regularity lemma for the graph summarization
    :param kind: the kind of implementation to generate. The currently accepted strings are 'alon' for the Alon
                 the Alon implementation, and 'frieze_kannan' for the Frieze and Kannan implementation
    :param sim_mat: the similarity matrix representing the graph
    :param epsilon: the epsilon parameter to determine the regularity of the partition
    :param is_weighted: set it to True to specify if the graph is weighted
    :param random_initialization: set it to True to perform to generate a random partition of the graph
    :param random_refinement: set it to True to randomly re-asset the nodes in the refinement step
    :param is_fully_connected_reduced_matrix: if set to True the similarity matrix is not thresholded and a fully
           connected graph is generated
    :param is_no_condition_considered_regular: if set to True, when no condition
    :return:
    """
    alg = srl.SzemerediRegularityLemma(sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs)

    if random_initialization:
        alg.partition_initialization = partition_initialization.random
    else:
        alg.partition_initialization = partition_initialization.degree_based

    if random_refinement:
        alg.refinement_step = refinement_step.random
    else:
        alg.refinement_step = refinement_step.degree_based

    if kind == "alon":
        alg.conditions = [conditions.alon1, conditions.alon2, conditions.alon3]
    elif kind == "frieze_kannan":
        alg.conditions = [conditions.frieze_kannan]
    else:
        raise ValueError("Could not find the specified graph summarization method")

    return alg
