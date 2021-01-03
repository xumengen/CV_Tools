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
from functional import *


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

    # print(solver.compute_depth_of_scence_point([50, 50, 0], [25, 50, 1], velocity=0.1, move_method='x-axis', pixel_size=0.1, focal_length=35))
    # print(solver.compute_depth_of_scence_point([50, 70, 0], [45, 63, 1], velocity=0.1, move_method='z-axis',center_coordinate=[100, 140]))

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

    # feature_vector_array = [[[5, 10, 15], [10, 15, 30], [10, 10, 25]], [[10, 10, 15], [5, 20, 15], [10, 5, 30]], [[5, 5, 15], [30, 10, 5], [30, 10, 10]]]
    # solver.region_growing(feature_vector_array)
    # solver.region_merge(feature_vector_array)
    # solver.k_means(feature_vector_array, 2, [[5, 10, 15], [10, 10, 25]])
    # solver.region_split_and_merge(feature_vector_array)
    # solver.agglomerative_hierarchical_clustering(feature_vector_array, k=4)

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

    # image_region = [[1, 0, 0, 1],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0]]
    # theta_1 = [0, 45, 90, 135]
    # theta_2 = [0, 30, 60, 90, 120, 150]
    # print(solver.hough_transform(image_region, theta_1))
    # print(solver.hough_transform(image_region, theta_2))

    # result = solver.compute_thin_lens_equation(f=35, z1=3000, z2=None)
    # print(result)
    # result = solver.compute_thin_lens_equation(f=35, z1=500, z2=None)
    # print(result)

    # result = solver.compute_3d_point_2d_coordinate([10, 10, 500], [244, 180], [925, 740])
    # print(result)

    # ori_imgae = [[[205, 195],
    #               [238, 203]],
    #              [[143, 138],
    #               [166, 143]],
    #              [[154, 145],
    #               [174, 151]],
    # ]
    # result = solver.convert_rbg_to_gray(ori_imgae, 8)
    # print(result)
    # result = solver.convert_rbg_to_gray(ori_imgae, 2)
    # print(result)

    # mask = [[1, 0], [1, 1]]
    # I1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # I2 = [[0, 0, 0], [1, 1, 0], [0, 1, 0]]
    # result_1 = solver.convolution(mask, I1)
    # result_2 = solver.convolution(mask, I2)
    # print(result_1)
    # print(result_2)

    # mask = [[0, 0, 0],
    #         [0, 0 ,1],
    #         [0, 0, 0]]
    # I = [[0.25, 1, 0.8],
    #      [0.75, 1, 1],
    #      [0, 1, 0.4]]
    # result = solver.convolution(mask, I, 'same')
    # print(result)

    # mask = [[1, 0.5, 0.1],
    #         [0.5, 0.25, 0.05],
    #         [0.1, 0.05, 0.01]]
    # I = [[1, 1, 1],
    #      [1, 1, 1],
    #      [1, 1, 1]]
    # print(solver.convolution(mask, I, 'same'))

    #print(solver.compute_pixel_val_using_gaussian((3, 3), standard_deviation=0.46))

    # tutorial9
    # template = [[100, 150, 200],
    #             [150, 10, 200],
    #             [200, 200, 250]]
    # I = [[60, 50, 40, 40],
    #      [150, 100, 100, 80],
    #      [50, 20, 200, 80],
    #      [200, 150, 150, 50]]
    # print(solver.find_object_location(template, I, 'normalised_cross_correlation'))
    # print(solver.find_object_location(template, I, 'sum_of_absolute_differences'))

    # template = [[1, 1, 1],
    #             [1, 0, 1],
    #             [1, 1, 1]]
    # I = [[0, 0, 0, 0],
    #      [1, 1, 1, 0],
    #      [0, 0, 1, 0],
    #      [1, 1, 1, 0]]
    # print(solver.find_object_location(template, I, 'minimum_distance'))

    # template_1 = [[1, 1, 1],
    #               [1, 0, 0],
    #               [1, 1, 1]]
    # template_2 = [[1, 0, 0],
    #               [1, 0, 0],
    #               [1, 1, 1]]
    # template_3 = [[1, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 1]]
    # image = [[1, 1, 1],
    #          [1, 0, 0],
    #          [1, 1, 1]]
    # print(solver.best_template_match([template_1, template_2, template_3], image, method='cross_correlation'))
    # print(solver.best_template_match([template_1, template_2, template_3], image, method='normalised_cross_correlation'))
    # print(solver.best_template_match([template_1, template_2, template_3], image, method='correlation_coefficient'))
    # print(solver.best_template_match([template_1, template_2, template_3], image, method='SAD'))

    # image_1 = [2,0,0,5,1,0,0,0,3,1]
    # image_2 = [0,0,1,2,0,3,1,0,1,0]
    # image_3 = [1,1,2,0,0,1,0,3,1,1]
    # new_image = [2,1,1,0,1,1,0,2,0,1]
    # print(solver.best_template_match([image_1, image_2, image_3], new_image, method='normalised_cross_correlation'))

    p1 = [40,-40,400]
    p2 = [23.3,-6.7,483.3]
    p3 = [15,10,525]
    p4 = [-10,60,650]
    print(solver.compute_cross_ratio(p1, p2, p3, p4, method='3d'))
    print(solver.compute_cross_ratio(p1, p2, p3, p4, method='2d', center_coordinate=[244,180], magnification_factors=[925, 740]))
