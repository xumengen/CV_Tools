# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/12/31 00:00:00
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import os
import numpy as np
from tutorial import *


if __name__ == '__main__':
    solver = Tutorail_solver()

    # result = solver.compute_distance_between_point_and_horopter(400, 60, -15, 15)
    # print(result)

    # patch_1 = [[190, 200, 90, 110, 90], [190, 200, 90, 110, 90]]
    # patch_2 = [[110, 170, 160, 70, 70], [110, 170, 160, 70, 70]]
    # patch_3 = [[100, 60, 170, 200, 90], [100, 60, 170, 200, 90]]
    # patch_4 = [[90, 100, 100, 190, 190], [90, 100, 100, 190, 190]]
    # result_1, result_2 = solver.compute_segment_moving_object_from_background([patch_1, patch_2, patch_3, patch_4], 50, 0.5)
    # print("the result of image difference is \n{},\nthe result of background substraction is \n{}".format(result_1, result_2))

    # class_list = ['A', 'A', 'B', 'B']
    # feature_vector_list = [[7, 7], [7, 4], [3, 4], [1, 4]]
    # object_feature_vector = [3, 7]
    # result_1, result_2, result_3 = solver.compute_object_class(class_list, feature_vector_list, object_feature_vector, k=3)
    # print("the result of nearest mean classifier is {}.\nthe result of nearest neighbour classifier is {}.\nthe result ofk-nearest neighbour classifier is {}.".format(result_1, result_2, result_3))
    
    # left_image = [[4, 7, 6, 7], [3, 4, 5, 4], [8, 7, 6, 8]]
    # right_image = [[7, 6, 7, 5], [4, 5, 4, 5], [7, 6, 8, 7]]
    # coordinate = [2, 3]
    # result = solver.compute_similarity_between_point_and_image(coordinate, left_image, right_image)
    # print(result)

    # Ix = [[1, 0, 0, 0, 0], [-1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    # Iy = [[1, -1, 0, 0, 0], [-1, 0, -1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
    # result = solver.compute_harris_corner_detector(Ix, Iy)
    # print(result)

    feature_vector_array = [[[5, 10, 15], [10, 15, 30], [10, 10, 25]], [[10, 10, 15], [5, 20, 15], [10, 5, 30]], [[5, 5, 15], [30, 10, 5], [30, 10, 10]]]
    # solver.region_growing(feature_vector_array)
    # solver.region_merge(feature_vector_array)
    # solver.k_means(feature_vector_array, 2, [[5, 10, 15], [10, 10, 25]])
    # solver.region_split_and_merge(feature_vector_array)
    solver.agglomerative_hierarchical_clustering(feature_vector_array, k=4)

    # input_array = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 0, 1, 1, 1, 0, 0],
    #                [0, 0, 0, 0, 1, 1, 1, 0, 0],
    #                [0, 0, 1, 1, 0, 1, 1, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 0, 0],]

    # output_array = solver.dilation(input_array, 'hvd')
    # print(output_array)
    # output_array = solver.erosion(output_array, 'hvd')
    # print(output_array)

    # output_array = solver.erosion(input_array, 'hv')
    # print(output_array)
    # output_array = solver.dilation(output_array, 'hv')
    # print(output_array)
