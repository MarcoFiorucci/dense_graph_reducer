import numpy as np
import os
import matplotlib.pyplot

img_dir_path = "..//test_data//images"
img_res_path = "..//test_data//results//resize_factor_0_25"
alon_epsilons = np.arange(0.5, 0.8, 0.05)
FK_epsilons = np.arange(0.2, 0.5, 0.05)

if __name__ == "__main__":
    one_stage = False
    two_stage = True
    if one_stage:
        for img_path in sorted(os.listdir(os.path.normpath(img_dir_path))):
            img_name = os.path.splitext(img_path)[0]
            ar = np.load(os.path.normpath(img_res_path + "//one_stage" + "//" + img_name + "//" + img_name + ".npy"))
            matplotlib.pyplot.matshow(ar)
            matplotlib.pyplot.show()

    if two_stage:
        for img_path in sorted(os.listdir(os.path.normpath(img_dir_path))):
            for eps in alon_epsilons:
                img_name = os.path.splitext(img_path)[0]
                if os.path.exists(os.path.normpath(img_res_path + "//two_stage" + "//alon//" + img_name + "//" + img_name + "_eps_" + str.replace(format(eps, '.2f'), ".", "_") + ".npy")):
                    ar = np.load(os.path.normpath(img_res_path + "//two_stage" + "//alon//" + img_name + "//" + img_name + "_eps_" + str.replace(format(eps, '.2f'), ".", "_") + ".npy"))
                    matplotlib.pyplot.matshow(ar)
                    matplotlib.pyplot.show()


