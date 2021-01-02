# -*- encoding: utf-8 -*-
'''
@File    :   tutorial.py
@Time    :   2020/12/25 00:00:00
@Author  :   XuMengEn 
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''

import os
import numpy as np
import math
import argparse
from collections import defaultdict

class Tutorail_solver:
    def __init__(self):
        pass
 
    def compute_distance_between_point_and_horopter(self, baseline_length, angle_z_baseline, a_l, a_r):
        """ compute the distance between the point and the horopter
        
        Args:
        baseline_length: baseline between the camera centers
        angle_z_baseline: angle between the baseline and z-axes(degree)
        a_l, a_r: the line-of-sight of point with z-axees of left and right camera respectively
        
        Returns:
        the distance between the point and the horopter
        """
        assert abs(a_l) == abs(a_r)
        from math import tan
        from math import pi
        from math import radians
        if a_l == a_r:
            return 0

        angle_1 = radians(angle_z_baseline)
        distance_fix_and_baseline = 0.5 * baseline_length * tan(angle_1)
        angle_2 = radians(angle_z_baseline + a_l)
        distance_point_and_baseline = 0.5 * baseline_length * tan(angle_2)
        return round(abs(distance_point_and_baseline - distance_fix_and_baseline), 3)
    
    def compute_segment_moving_object_from_background(self, pixel_patches, thres, beta, method='both'):
        """ segment the moving object from background
     
        Args: 
        pixel_patches: the patch taken from frames of video
        thres: the threshold between background and object
        beta: the parameter in update formula
        method: segment method [image difference, background substraction, both]
        
        Returns:
        segment result
        """

        # image difference
        def image_diff():
            diff_array = abs(pixel_array[:-1] - pixel_array[1:])
            result = np.where(diff_array>thres, 1, 0)            
            return result

        # background substraction
        def bg_sub():
            bg_array = np.empty(pixel_array[0].shape)
            result = np.empty(pixel_array.shape, dtype=int)
            for i in range(len(pixel_array)):
                bg_array = (1 - beta) * bg_array + beta * pixel_array[i]
                diff_array = abs(pixel_array[i] - bg_array)
                result[i] = np.where(diff_array>thres, 1, 0)
            return result
            
        pixel_array = np.array(pixel_patches)
        if method == 'image_difference':
            return image_diff()

        elif method == 'background substraction':
            return bg_sub()

        else:
            result_1 = image_diff()
            result_2 = bg_sub()
            # return np.concatenate((result_1, result_2), axis=0)
            return result_1, result_2

    def compute_object_class(self, class_list, feature_vector_list, object_feature_vector, k):
        """ compute the category of the object
    
        Args:
        class_list: the list contain the object category ['A', 'A', 'B', 'B']
        feature_vetor_list: the list contain the feature vetor
        object_feature_vector: the feature vector of the object
        """
        
        # nearest mean classifier
        def nearest_mean_classifier():
            record_dict = dict()
            for idx, class_info in enumerate(class_list):
                if class_info not in record_dict.keys():
                    record_dict[class_info] = [feature_vector_list[idx]]
                else:
                    record_dict[class_info].append(feature_vector_list[idx])
            dist = float('inf')
            min_class = None
            for key, val in record_dict.items():
                mean = np.mean(np.array(val), axis=0)
                if self.compute_eucli_distance(mean, np.array(object_feature_vector)) < dist:
                    dist = self.compute_eucli_distance(mean, np.array(object_feature_vector))
                    min_class = key
            return min_class
 
        # nearest neighbour classifier
        def nearest_neighbour_classifier():
            dist_list = compute_dist_list()
            return class_list[dist_list.index(min(dist_list))]

        # k-nearest neighbour classifier
        def k_nearest_neighbour_classifier():
            dist_list = compute_dist_list()
            dist_array = np.array(dist_list)
            res_array = np.argsort(dist_array)[:k]
            result_dict = defaultdict(int)
            for i in res_array:
                result_dict[class_list[i]] += 1
            max_val, max_class = 0, None
            for key, val in result_dict.items():
                if val > max_val:
                    max_val = val
                    max_class = key
            return max_class
        
        def compute_dist_list():
            dist_list = []
            for idx, feature_vector in enumerate(feature_vector_list):
                dist = self.compute_eucli_distance(np.array(feature_vector), np.array(object_feature_vector))
                dist_list.append(dist)
            return dist_list
        
        result_1 = nearest_mean_classifier()
        result_2 = nearest_neighbour_classifier()
        result_3 = k_nearest_neighbour_classifier()
        return result_1, result_2, result_3

    def compute_eucli_distance(self, p1, p2):
        """compute the eulic distance between two vectors
        """
        return np.sqrt(np.sum(np.square(p1 - p2)))

    def compute_SAD_diff(self, array1, array2):
        """
        """
        return np.sum(np.absolute(array1 - array2))

    def compute_similarity_between_point_and_image(self, coordinate, left_image, right_image, k=3):
        """
        """
        assert k % 2 != 0
        left_array = np.array(left_image)
        right_array = np.array(right_image)
        interval = (k - 1) // 2
        ori_array = left_array[coordinate[0]-1-interval:coordinate[0]+interval, coordinate[1]-1-interval:coordinate[1]+interval]
        compare_array = np.pad(right_array, ((interval, interval), (interval, interval)), 'constant', constant_values=(0, 0))
        result_array = np.empty(left_array.shape)
        for i in range(len(right_array)):
            for j in range(len(right_array[0])):
                compare_sub_array = compare_array[i:k+i, j:k+j]
                val = self.compute_SAD_diff(ori_array, compare_sub_array)
                result_array[i][j] = val
        return result_array
   
    def compute_harris_corner_detector(self, Ix, Iy, k=0.05, length=3):
        """
        """
        Ix_array = np.array(Ix)
        Iy_array = np.array(Iy)
        Ix_1 = np.power(Ix_array, 2)
        Iy_1 = np.power(Iy_array, 2)
        IxIy = Ix_array * Iy_array

        Ix_2 = self.slide_window_operation(Ix_array, Ix_array, length)
        Iy_2 = self.slide_window_operation(Iy_array, Iy_array, length)
        IxIy_2 = self.slide_window_operation(Ix_array, Iy_array, length)

        IxIy_3 = Ix_2 * Iy_2
        IxIy_4 = np.power(IxIy_2, 2)
        IxIy_5 = np.power(Ix_2+Iy_2, 2)
        R = (IxIy_3 - IxIy_4) - k * IxIy_5
        return R
        
    def slide_window_operation(self, array_1, array_2, length=3):
        """
        """
        assert array_1.shape == array_2.shape
        interval = (length - 1) // 2
        pad_array_1 = np.pad(array_1, ((interval, interval), (interval, interval)), 'constant', constant_values=(0, 0))
        pad_array_2 = np.pad(array_2, ((interval, interval), (interval, interval)), 'constant', constant_values=(0, 0))
        result_array = np.empty(array_1.shape)
        for i in range(len(array_1)):
            for j in range(len(array_1[0])):
                result_array[i][j] = np.sum(pad_array_1[i:length+i, j:length+j] * pad_array_2[i:length+i, j:length+j])
        return result_array
        
    def region_growing(self, feature_vector_array, method='SAD', thres=12, mode='hvd', start=(0,0)):
        """
        """
        
        def sub_region_growing(i, j):
            neighbour_array = self.find_neighbour_array(i, j, visited_array, result_array.shape, mode)
            neighbour_array = self.assign_neighbour_value(i, j, result_array, feature_vector_array, visited_array, neighbour_array, method, thres)
            for coordinate in neighbour_array:
                sub_region_growing(coordinate[0], coordinate[1])
        
        def check_result_array():
            for i in range(len(result_array)):
                for j in range(len(result_array[0])):
                    if result_array[i][j] < 1:
                        return [i, j]
            return None

        feature_vector_array = np.array(feature_vector_array)
        result_array = np.empty(feature_vector_array.shape[:-1])
        visited_array = [[False for j in range(feature_vector_array.shape[1])] for i in range(feature_vector_array.shape[0])]
        i, j = start[0], start[1]
        label = 1 
        result_array[i][j] = label
        sub_region_growing(i, j)
        
        while check_result_array():
            label += 1
            [i, j] = check_result_array()
            result_array[i][j] = label
            sub_region_growing(i, j)    
    
        print("the result of region growing is\n {}\n".format(result_array))
        return result_array

    def find_neighbour_array(self, i, j, visited_array, boundary, mode):
        """
        """
        assert mode in ['hvd', 'hv']
        if mode == 'hvd':
            neighbour = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
            neighbour_sub = np.array([i, j]) + neighbour
            neighbour_list = list()
            for val in neighbour_sub:
                if 0 <= val[0] < boundary[0] and 0 <= val[1] < boundary[1] and visited_array[val[0]][val[1]] == False:
                    neighbour_list.append(val)
        elif mode == 'hv':
            neighbour = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])
            neighbour_sub = np.array([i, j]) + neighbour
            neighbour_list = list()
            for val in neighbour_sub:
                if 0 <= val[0] < boundary[0] and 0 <= val[1] < boundary[1] and visited_array[val[0]][val[1]] == False:
                    neighbour_list.append(val)
        return np.array(neighbour_list)

    # TODO merge v1v2
    def find_neighbour_array_v2(self, i, j, boundary, mode):
        """
        """
        assert mode in ['hvd', 'hv']
        if mode == 'hvd':
            neighbour = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
            neighbour_sub = np.array([i, j]) + neighbour
            neighbour_list = list()
            for val in neighbour_sub:
                if 0 <= val[0] < boundary[0] and 0 <= val[1] < boundary[1]:
                    neighbour_list.append(val)
        elif mode == 'hv':
            neighbour = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])
            neighbour_sub = np.array([i, j]) + neighbour
            neighbour_list = list()
            for val in neighbour_sub:
                if 0 <= val[0] < boundary[0] and 0 <= val[1] < boundary[1]:
                    neighbour_list.append(val)
        return np.array(neighbour_list)
             
    def assign_neighbour_value(self, i, j, result_array, feature_vector_array, visited_array, neighbour_array, method, thres, region_method='growing'):
        """
        """
        #TODO 'L2'
        assert method in ['SAD']
        result_list = list()
        if region_method == 'growing':
            compare_feature_vector = feature_vector_array[i][j]
        elif region_method == 'merge':
            region_label = result_array[i][j]
            correspond_index = np.where(result_array==region_label)
            compare_feature_vector = np.mean(feature_vector_array[correspond_index], axis=0)
        for coordinate in neighbour_array:
            feature_vector = feature_vector_array[coordinate[0]][coordinate[1]]
            if method == 'SAD':
                distance = self.compute_SAD_diff(compare_feature_vector, feature_vector)
                if distance <= thres:
                    result_array[coordinate[0]][coordinate[1]] = result_array[i][j]
                    visited_array[coordinate[0]][coordinate[1]] = True
                    result_list.append(coordinate)
        return np.array(result_list)

    def region_merge(self, feature_vector_array, method='SAD', thres=12, mode='hvd', start=(0,0), result_array=[]):  
        """
        """

        def sub_region_merge(i, j):
            neighbour_array = self.find_neighbour_array(i, j, visited_array, result_array.shape, mode)
            neighbour_array = self.assign_neighbour_value(i, j, result_array, feature_vector_array, visited_array, neighbour_array, method, thres, region_method='merge')
            for coordinate in neighbour_array:
                sub_region_merge(coordinate[0], coordinate[1])
        
        def check_result_array():
            for i in range(len(visited_array)):
                for j in range(len(visited_array[0])):
                    if not visited_array[i][j]:
                        return [i, j]
            return None

        assert len(np.array(feature_vector_array).shape) == 3
        feature_vector_array = np.array(feature_vector_array)
        if type(result_array) == list:
            result_array = np.array([i+1 for i in range(feature_vector_array.shape[0]*feature_vector_array.shape[1])]).reshape(np.array(feature_vector_array).shape[:-1])
        else:
            result_array = result_array
        visited_array = [[False for j in range(feature_vector_array.shape[1])] for i in range(feature_vector_array.shape[0])]
        i, j = start[0], start[1]
        visited_array[i][j] = True
        sub_region_merge(i, j)
        
        while check_result_array():
            [i, j] = check_result_array()
            visited_array[i][j] = True
            sub_region_merge(i, j)    
    
        print("the result of region merge is\n {}\n".format(result_array))
        return result_array

    # TODO change 3*3 to n*n
    def region_split_and_merge(self, feature_vector_array, method='SAD', thres=12, mode='hvd', start=(0, 0)):
        """
        """

        def change_result_array(sub_feature_vector_array, label):
            unit_param = int(len(correspond_index[0]) ** 0.5)
            old_array_shape = [unit_param, unit_param, feature_vector_array.shape[-1]]
            if unit_param % 2 != 0:
                new_array_shape = [unit_param+1, unit_param+1, feature_vector_array.shape[-1]]
                sub_feature_vector_array = sub_feature_vector_array.reshape(old_array_shape)
                new_sub_feature_vector_array = np.zeros(new_array_shape)
                for i in range(new_array_shape[0]):
                    for j in range(new_array_shape[1]):
                        if 0 <= i < old_array_shape[0] and j == old_array_shape[1]:
                            new_sub_feature_vector_array[i][j][:] = sub_feature_vector_array[i][j-1][:]
                        elif 0 <= j < old_array_shape[1] and i == old_array_shape[0]:
                            new_sub_feature_vector_array[i][j][:] = sub_feature_vector_array[i-1][j][:]
                        elif i == old_array_shape[0] and j == old_array_shape[1]:
                            new_sub_feature_vector_array[i][j][:] = sub_feature_vector_array[i-1][j-1][:]
                        else:
                            new_sub_feature_vector_array[i][j][:] = sub_feature_vector_array[i][j][:]
                sub_feature_vector_array = new_sub_feature_vector_array
            else:
                new_array_shape = [unit_param, unit_param, feature_vector_array.shape[-1]]
            new_result_array = np.zeros(new_array_shape[:-1])
            partition_param = (unit_param + 1) // 2
            for i in range(new_result_array.shape[0]):
                for j in range(new_result_array.shape[1]):
                    if (i+1) <= partition_param and (j+1) <= partition_param:
                        new_result_array[i][j] = label
                    elif (i+1) <= partition_param and (j+1) > partition_param:
                        new_result_array[i][j] = count + 1
                    elif (i+1) > partition_param and (j+1) <= partition_param:
                        new_result_array[i][j] = count + 2
                    else:
                        new_result_array[i][j] = count + 3
            if unit_param % 2 != 0:
                return sub_feature_vector_array, new_result_array
            else:
                return new_result_array


        feature_vector_array = np.array(feature_vector_array)
        assert feature_vector_array.shape[0] == feature_vector_array.shape[1] == 3
        result_array = np.ones(feature_vector_array.shape[:-1])
        count = 1
        correspond_index = np.where(result_array==1)
        sub_feature_vector_array = feature_vector_array[correspond_index]
        for i in range(sub_feature_vector_array.shape[0]):
            for j in range(i+1, sub_feature_vector_array.shape[0]):
                if method == 'SAD':
                    dist = self.compute_SAD_diff(sub_feature_vector_array[i], sub_feature_vector_array[j])
                    if dist > 12:
                        feature_vector_array, result_array = change_result_array(sub_feature_vector_array, 1)    
                        count += 3
                        break
            break
        for label in [1, 2, 3, 4]:
            correspond_index = np.where(result_array==label)
            sub_feature_vector_array = feature_vector_array[correspond_index]
            for i in range(sub_feature_vector_array.shape[0]):
                for j in range(i+1, sub_feature_vector_array.shape[0]):
                    if method == 'SAD':
                        dist = self.compute_SAD_diff(sub_feature_vector_array[i], sub_feature_vector_array[j])
                        if dist > 12:
                            new_result_array = change_result_array(sub_feature_vector_array, label)
                            correspond_index_array = np.array(list(zip(correspond_index[0], correspond_index[1])))
                            index = 0
                            for m in range(new_result_array.shape[0]):
                                for n in range(new_result_array.shape[1]):
                                    point = correspond_index_array[index]
                                    index += 1
                                    result_array[point[0]][point[1]] = new_result_array[m][n]
                            count += 3
                            break
                break

        for i in range(result_array.shape[0]):
            for j in range(result_array.shape[1]):
                label = result_array[i][j]
                correspond_index = np.where(result_array==label)
                if len(correspond_index[0]) > 1:
                    mean_feature_vector = np.mean(feature_vector_array[correspond_index], axis=0)
                    for i in range(len(correspond_index[0])):
                        feature_vector_array[correspond_index[0][i]][correspond_index[1][i]] = mean_feature_vector

        print("the result of region split is\n {}\n".format(result_array))
        print("the result feature vector of regoin split is\n {}\n".format(feature_vector_array))
            
                            
        result_array = self.region_merge(feature_vector_array, method=method, thres=thres, mode=mode, start=start, result_array=result_array)
        print("the result of region split and merge is\n {}\n".format(result_array))
        return result_array

    def k_means(self, feature_vector_array, k, ori_feature_vetor_array, method='SAD'):
        """
    
        """
        def compute_new_cluster_center():
            new_ori_feature_vetor_array = np.zeros(ori_feature_vetor_array.shape)
            for i in range(1, k+1):
                correspond_index = np.where(result_array==i)
                new_ori_feature_vetor_array[i-1][:] = np.mean(feature_vector_array[correspond_index], axis=0)
            return new_ori_feature_vetor_array

        assert k > 0
        assert k == len(ori_feature_vetor_array)
        feature_vector_array = np.array(feature_vector_array)
        ori_feature_vetor_array = np.array(ori_feature_vetor_array)
        result_array = np.empty(feature_vector_array.shape[:-1])

        while True:
            result_array_copy = result_array.copy()
            for i in range(feature_vector_array.shape[0]):
                for j in range(feature_vector_array.shape[1]):
                    feature_vector = feature_vector_array[i][j]
                    assert feature_vector.shape == ori_feature_vetor_array[0].shape
                    dist_list = list()
                    for m in range(k):
                        dist_list.append(self.compute_SAD_diff(feature_vector, ori_feature_vetor_array[m]))
                    label = dist_list.index(min(dist_list)) + 1
                    result_array[i][j] = label
            if (result_array_copy == result_array).all():
                break
            ori_feature_vetor_array = compute_new_cluster_center()

        print("the result of region k-means is\n {}\n".format(result_array))
        return result_array

    # TODO add other cluster method
    def agglomerative_hierarchical_clustering(self, feature_vector_array, k=3, method='SAD', cluster_method='centroid'):
        """
        """

        feature_vector_array = np.array(feature_vector_array)
        cluster_array = feature_vector_array.reshape(-1, feature_vector_array.shape[-1])

        record_list = []
        for i in range(1, cluster_array.shape[0]+1):
            record_list.append([i])
        while cluster_array.shape[0] > k:
            result_list = []
            for i in range(cluster_array.shape[0]):
                cluster_1 = cluster_array[i]
                for j in range(i+1, cluster_array.shape[0]):
                    cluster_2 = cluster_array[j]
                    dist = self.compute_SAD_diff(cluster_1, cluster_2)
                    result_list.append([i+1, j+1, dist])
            min_dist = min([i[-1] for i in result_list])
            pair_result = []
            for result in result_list:
                if result[-1] <= min_dist:
                    min_dist = result[-1]
                    pair_result.append(result[:-1])
            cluster_result = list()
            symbol = True
            for i in range(1, cluster_array.shape[0]+1):
                tmp = []
                for pair in pair_result:
                    if i in pair:
                        for val in pair:
                            tmp.append(val)
                for result in cluster_result:
                    if set(result) >= set(tmp):
                        symbol = False
                if symbol:
                    cluster_result.append(list(set(tmp)))
                symbol = True
                        
            new_cluster_index = list()
            for result in cluster_result:
                for i in result:
                    new_cluster_index.append(i)
            new_cluster_array = list()
            new_record_list = list()
            for i in range(len(cluster_result)):
                if not cluster_result[i]:
                    continue
                tmp = []
                for result in cluster_result[i]:
                    target = record_list[result-1]
                    for m in target:
                        tmp.append(m)
                new_record_list.append(tmp)
                index = np.array(cluster_result[i]) - 1
                if cluster_method == 'centroid':
                    new_cluster_array.append(np.mean(cluster_array[index], axis=0))
            for i in range(1, cluster_array.shape[0]+1):
                if i in new_cluster_index:
                    continue
                else:
                    new_cluster_array.append(cluster_array[i-1])
                new_record_list.append(record_list[i-1])
            cluster_array = np.array(new_cluster_array)
            record_list = new_record_list

        print("the result of agglomerative hierarchical clustering is\n {}\n".format(record_list))
        return record_list

    def dilation(self, input_array, mode):
        """
        """
        input_array = np.array(input_array)
        assert len(input_array.shape) == 2
        output_array = np.zeros(input_array.shape)
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                if input_array[i][j] == 0:
                    neighbour_array = self.find_neighbour_array_v2(i, j, input_array.shape, mode)
                    for neighbour in neighbour_array:
                        if input_array[neighbour[0]][neighbour[1]] == 1:
                            output_array[i][j] = 1
                            break
                else:
                    output_array[i][j] = input_array[i][j]
        return output_array

    def erosion(self, input_array, mode):
        """
        """
        input_array = np.array(input_array)
        assert len(input_array.shape) == 2
        output_array = np.ones(input_array.shape)
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                if input_array[i][j] == 1:
                    neighbour_array = self.find_neighbour_array_v2(i, j, input_array.shape, mode)
                    for neighbour in neighbour_array:
                        if input_array[neighbour[0]][neighbour[1]] == 0:
                            output_array[i][j] = 0
                            break
                else:
                    output_array[i][j] = input_array[i][j]
        return output_array

    def compute_thin_lens_equation(self, f=None, z1=None, z2=None):
        """
        thin lens equation: 1/f = 1/z1 + 1/z2
        """
        assert f > 0
        if not f and z1 and z2:
            return 1.0 / (1.0 / abs(z1) + 1.0 / abs(z2))
        elif f and not z1 and z2:
            return 1.0 / (1.0 / abs(f) - 1.0 / abs(z2))
        elif f and z1 and not z2:
            return 1.0 / (1.0 / abs(f) - 1.0 / abs(z1))
        else:
            print("Your input kidding me!!!")

    def compute_3d_point_2d_coordinate(self, ori_coordinate, image_principal_point, magnification_factors):
        """
        """
        assert len(ori_coordinate) == 3
        assert len(image_principal_point) == 2
        assert len(magnification_factors) == 2
        array_1 = np.array([[magnification_factors[0], 0, image_principal_point[0]],
                            [0, magnification_factors[1], image_principal_point[1]],
                            [0, 0, 1]])
        array_2 = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])
        array_3 = np.array(ori_coordinate+[1.0])
        result_array = (1 / float(ori_coordinate[-1]) * np.dot(np.dot(array_1, array_2), array_3)).tolist()
        new_result_array = list()
        for result in result_array:
            new_result_array.append(int(round(result)))
        return new_result_array

    def convert_rbg_to_gray(self, ori_image, bit=8):
        """
        """
        assert len(ori_image) == 3
        ori_image_array = np.array(ori_image)
        result_image = np.mean(ori_image_array, axis=0)
        new_result_image = np.zeros(result_image.shape)
        for i in range(result_image.shape[0]):
            for j in range(result_image.shape[1]):
                new_result_image[i][j] = int(round(result_image[i][j]))
        
        interval = (2 ** 8) / (2 ** bit)
        final_result = np.zeros(new_result_image.shape)
        for i in range(new_result_image.shape[0]):
            for j in range(new_result_image.shape[1]):
                final_result[i][j] = new_result_image[i][j] // interval

        return final_result

    def convolution(self, mask, I, method='inside'):
        """
        """
        mask = np.array(mask)
        I = np.array(I)
        assert len(I.shape) == 2
        mask = np.rot90(mask, 2)
        if method == 'inside':
            out_dimension_1 = I.shape[0] - mask.shape[0] + 1
            out_dimension_2 = I.shape[1] - mask.shape[1] + 1
            result_array = np.zeros((out_dimension_1, out_dimension_2))
            for i in range(out_dimension_1):
                for j in range(out_dimension_2):
                    result_array[i][j] = np.sum(mask * I[i:i+mask.shape[0], j:j+mask.shape[1]])
        elif method == 'same':
            assert mask.shape[0] % 2 != 0
            assert mask.shape[1] % 2 != 0
            out_dimension_1 = I.shape[0]
            out_dimension_2 = I.shape[1]
            result_array = np.zeros((out_dimension_1, out_dimension_2))
            pad_size_1 = (mask.shape[0] - 1) // 2
            pad_size_2 = (mask.shape[1] - 1) // 2
            new_I = np.pad(I, ((pad_size_1, pad_size_1), (pad_size_2, pad_size_2)), 'constant', constant_values=(0, 0))
            for i in range(out_dimension_1):
                for j in range(out_dimension_2):
                    result_array[i][j] = np.sum(mask * new_I[i:i+mask.shape[0], j:j+mask.shape[1]])
        
        return result_array

    def compute_pixel_val_using_gaussian(self, array_size, standard_deviation, decimal=2):
        """
        """
        assert len(array_size) == 2
        assert array_size[0] == array_size[1]
        shift = (array_size[0] - 1) // 2
        result_array = np.zeros(array_size)
        for i in range(array_size[0]):
            for j in range(array_size[1]):
                x = i - shift
                y = j - shift
                result_array[i][j] = round(math.exp(-(x**2+y**2)/(2*standard_deviation**2)) / (2*math.pi*(standard_deviation**2)), decimal)
        return result_array
        

