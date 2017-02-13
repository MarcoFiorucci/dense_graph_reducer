from szemeredi_utils import generate_szemeredi_segs

from BerkeleyImgSegmentationPerfEvaluator import BSD_utils
from BerkeleyImgSegmentationPerfEvaluator import BSD_perfomance_evaluation

root = ".."
main_dir = root + "//test_data"
img_dir_path = root + "//test_data//images"
gt_dir_path = root + "//test_data//groundTruth"
alg_kinds = [# 'alon',
             'frieze_kannan']
res_dir_path = root + "//test_data//results/resize_factor_0_25"

if __name__ == "__main__":
    BSD_utils.create_isoF_figure()
    BSD_perf_one_stage = BSD_perfomance_evaluation.BSDPerfomanceEvaluation(img_dir_path, gt_dir_path, generate_szemeredi_segs)

    BSD_perf_one_stage.compute_measures(res_dir_path + "//DS_n//one_stage")
    print(BSD_perf_one_stage.RIs_and_VOIs)
    # BSD_utils.plot_precision_recall_curve(BSD_perf_one_stage.scores)
    # input("Waiting for a key...")
    print("one stage: F1 measure = " + str(BSD_perf_one_stage.compute_F1_measure()))

    BSD_perf_two_stage = BSD_perfomance_evaluation.BSDPerfomanceEvaluation(img_dir_path, gt_dir_path,
                                                                           generate_szemeredi_segs)
    for alg_kind in alg_kinds:
        BSD_perf_two_stage.compute_measures(res_dir_path + "//DS_n//two_stage//" + alg_kind)
        print(BSD_perf_two_stage.RIs_and_VOIs)
        # BSD_utils.plot_precision_recall_curve(BSD_perf_two_stage.scores)
        print(alg_kind + " two stage: F1 measure = " + str(BSD_perf_two_stage.compute_F1_measure()))

