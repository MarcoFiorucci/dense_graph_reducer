function [reduced, reconstructed] = summarizegraph(adjMat, algKind, epsilon, isWeighted, reconstruct, thresh,
                                                   debugInfo, randomInitialization, randomRefinement,
                                                   dropEdgesBetweenIrregularPairs)
% SUMMARIZEGRAPH  Perform a graph summarization based on the Szemeredi Regularity Lemma.
%   REDUCED = SUMMARIZEGRAPH(ADJMAT, ALGKIND, EPSILON) reduce the adjacency matrix ADJMAT according to the specified
%                                                      ALGKIND summarization method and to the specified EPSILON
%                                                      parameter
%   REDUCED = SUMMARIZEGRAPH(ADJMAT, ALGKIND, EPSILON, thresh)
    ndarray = py.numpy.array(reshape(adjMat, 1, numel(adjMat))).reshape(size(adjMat, 1), size(adjMat, 2));

    alg = py.graph_reducer.szemeredi_lemma_builder.generate_szemeredi_reg_lemma_implementation(algKind, ndarray,
                                                                                               epsilon, is_weighted,
                                                                                               randomInitialization,
                                                                                               randomRefinement,
                                                                                        dropEdgesBetweenIrregularPairs);
    alg.run(pyargs('debug', debugInfo))
    reduced = py.getattr(alg, 'reduced_sim_mat');
    reducedShape = reduced.shape;
    reduced = reshape(double(py.array.array('d', py.numpy.nditer(reduced))), [int64(reducedShape{1}), int64(reducedShape{2})]);

    if reconstruct
        reconstructed = alg.reconstruct_original_mat(thresh);
        reconstructedShape = reconstructed.shape;
        reconstructed = reshape(double(py.array.array('d', py.numpy.nditer(reconstructed))), [int64(reconstructedShape{1}), int64(reconstructedShape{2})]);
    end
end
