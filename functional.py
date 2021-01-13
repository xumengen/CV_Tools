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
 
    # tutorial 7_1
    def compute_z_from_two_coplanar_camera(self, f, B, left_coordinate, right_coordinate, pixel_size, coplanar='x-axis', decimal=2):
        """
        """
        if coplanar ==  'x-axis':
            return round((f * B) / (pixel_size * (left_coordinate[0] - right_coordinate[0])), decimal)
        elif coplanar == 'y-axis':
            return round((f * B) / (pixel_size * (left_coordinate[1] - right_coordinate[1])), decimal)

    # tutorial 7_5
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
    
    # tutorial 8_4
    def compute_depth_of_scence_point(self, frame_1_point, frame_2_point, velocity, move_method='x-axis', pixel_size=None, focal_length=None, center_coordinate=None):
        """
        """
        if move_method == 'x-axis':
            assert focal_length
            assert pixel_size
            v_p = (frame_2_point[0]- frame_1_point[0]) / (frame_2_point[2] - frame_1_point[2]) * pixel_size
            return -focal_length * velocity / v_p
        elif move_method == 'z-axis':
            assert center_coordinate
            frame_1_point[0] = frame_1_point[0] - center_coordinate[0]
            frame_2_point[0] = frame_2_point[0] - center_coordinate[0]
            v_p = (frame_2_point[0] - frame_1_point[0]) / (frame_2_point[2] - frame_1_point[2])
            return frame_1_point[0] * velocity / v_p

    # tutorial 8_5
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
            bg_array = np.zeros(pixel_array[0].shape)
            result = np.zeros(pixel_array.shape, dtype=int)
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

    # tutorial 10_4
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

    # tutorial 6_4
    def compute_similarity_between_point_and_image(self, coordinate, left_image, right_image, k=3):
        """
        """
        assert k % 2 != 0
        coordinate = coordinate[::-1]
        left_array = np.array(left_image)
        right_array = np.array(right_image)
        interval = (k - 1) // 2
        ori_array = left_array[coordinate[0]-1-interval:coordinate[0]+interval, coordinate[1]-1-interval:coordinate[1]+interval]
        compare_array = np.pad(right_array, ((interval, interval), (interval, interval)), 'constant', constant_values=(0, 0))
        result_array = np.zeros(left_array.shape)
        for i in range(len(right_array)):
            for j in range(len(right_array[0])):
                compare_sub_array = compare_array[i:k+i, j:k+j]
                val = self.compute_SAD_diff(ori_array, compare_sub_array)
                result_array[i][j] = val
        return result_array
   
    # tutorial 6_9
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
        result_array = np.zeros(array_1.shape)
        for i in range(len(array_1)):
            for j in range(len(array_1[0])):
                result_array[i][j] = np.sum(pad_array_1[i:length+i, j:length+j] * pad_array_2[i:length+i, j:length+j])
        return result_array
        
    # tutorial 6_8
    def RANSAC(self, left_point, right_point, thres, trials, decimal=2):
        """
        """
        assert len(left_point) == len(right_point)
        left_point = np.array(left_point)
        right_point = np.array(right_point)
        result_list = []
        index_list = []
        for i in range(len(trials)):
            translation = left_point[i] - right_point[i]
            count = 0
            tmp_list = []
            for j in range(left_point.shape[0]):
                if j == i:
                    continue
                else:
                    new_right_point = left_point[j] - translation
                    dist = self.compute_SAD_diff(new_right_point, right_point[j])
                    if dist < thres: 
                        count += 1
                        tmp_list.append(j)
            result_list.append(count) 
            index_list.append(tmp_list)
            count = 0
        max_index = result_list.index(max(result_list))
        translation_list = []
        translation_list.append(left_point[max_index] - right_point[max_index])
        for i in index_list[max_index]:
            translation_list.append(left_point[i] - right_point[i])
        final_index_list = np.array([max_index]+index_list[max_index]) + 1
        print("the result of true correspondence is \n {}".format(final_index_list))
        return np.around(np.mean(np.array(translation_list), axis=0), decimal)

    # tutorial 5_6
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
        result_array = np.zeros(feature_vector_array.shape[:-1])
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
                if distance < thres:
                    result_array[coordinate[0]][coordinate[1]] = result_array[i][j]
                    visited_array[coordinate[0]][coordinate[1]] = True
                    result_list.append(coordinate)
        return np.array(result_list)

    # tutorial 5_8
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
    # tutorial 5_10
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

    # tutorial 5_12
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
        result_array = np.zeros(feature_vector_array.shape[:-1])

        count = 1
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
            print("the result of {}-th step is\n {}\n".format(count, result_array))
            if (result_array_copy == result_array).all():
                break
            count += 1
            ori_feature_vetor_array = compute_new_cluster_center()

        print("the result of region k-means is\n {}\n".format(result_array))
        return result_array

    def agglomerative_hierarchical_clustering(self, feature_vector_array, k=3, method='SAD', cluster_method='centroid'):
        """
        """

        
        assert cluster_method == 'centroid'
        feature_vector_array = np.array(feature_vector_array)
        cluster_array = feature_vector_array.reshape(-1, feature_vector_array.shape[-1])
        old_cluster_array = cluster_array.copy()

        record_list = []
        for i in range(1, cluster_array.shape[0]+1):
            record_list.append([i])
        step = 1
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
            for i in range(1, cluster_array.shape[0]+1):
                if i in new_cluster_index:
                    continue
                else:
                    new_record_list.append(record_list[i-1])
            for i in range(len(new_record_list)):
                cluster_index_list = np.array(new_record_list[i]) - 1
                if cluster_method == 'centroid':
                    new_cluster_array.append(np.mean(old_cluster_array[cluster_index_list], axis=0))
                else:
                    raise AttributeError("THIS METHOD IS NOR IMPLEMENTED!!!")
            cluster_array = np.array(new_cluster_array)
            record_list = new_record_list
            print("the feature of cluster of {}-th step is\n {}\n".format(step, cluster_array))
            print("the result of cluster of {}-th step is\n {}\n".format(step, record_list))
            step += 1

        print("the result of agglomerative hierarchical clustering is\n {}\n".format(record_list))
        return record_list

    # tutorial 6_5
    def compute_dist_between_two_feature_vector_array(self, array_1, array_2, method='SAD'):
        """
        Args:
        array_1: m1*n
        array_2: m2*n
        Return: m1*m2
        """

        array_1 = np.array(array_1)
        array_2 = np.array(array_2)
        assert len(array_1.shape) == 2 and len(array_2.shape) == 2
        result_array = list()
        for i in range(array_1.shape[0]):
            dist = list()
            for j in range(array_2.shape[0]):
                if method == 'SAD':
                    dist.append(self.compute_SAD_diff(array_1[i], array_2[j]))
                elif method == 'eucli':
                    dist.append(self.compute_eucli_distance(array_1[i], array_2[j]))
            result_array.append(dist[:])
        return np.array(result_array)

    # tutorial 5_15
    def agglomerative_hierarchical_clusteringv2(self, feature_vector_array, k=3, method='SAD', cluster_method='centroid'):
        """
        """

        feature_vector_array = np.array(feature_vector_array)
        cluster_array = feature_vector_array.reshape(-1, 1, feature_vector_array.shape[-1])
        cluster_list = cluster_array.tolist()
        old_cluster_array = cluster_array.copy()

        record_list = []
        for i in range(1, cluster_array.shape[0]+1):
            record_list.append([i])
        step = 1
        while len(record_list) > k:
            result_list = []
            for i in range(len(record_list)):
                cluster_1 = np.array(cluster_list[i])
                for j in range(i+1, len(record_list)):
                    cluster_2 = np.array(cluster_list[j])
                    if cluster_method == 'centroid':
                        dist = self.compute_SAD_diff(cluster_1, cluster_2)
                    elif cluster_method == 'single_link':
                        dist_array = self.compute_dist_between_two_feature_vector_array(cluster_1, cluster_2)
                        dist = np.min(dist_array)
                    elif cluster_method == 'complete_link':
                        dist_array = self.compute_dist_between_two_feature_vector_array(cluster_1, cluster_2)
                        dist = np.max(dist_array)
                    elif cluster_method == 'group_average':
                        dist_array = self.compute_dist_between_two_feature_vector_array(cluster_1, cluster_2)
                        dist = np.mean(dist_array)
                    result_list.append([i+1, j+1, dist])
            min_dist = min([i[-1] for i in result_list])
            cluster_result = []
            pair = [0, 0]
            for result in result_list:
                if result[-1] == min_dist:
                    pair[0] = result[0]
                    pair[1] = result[1]
                    symbol = False
                    for idx, cluster_info in enumerate(cluster_result):
                        if pair[0] in cluster_info or pair[1] in cluster_info:
                            cluster_info.extend([pair[0], pair[1]])
                            cluster_result[idx] = list(set(cluster_info))
                            symbol = True
                            break
                    if not symbol:
                        cluster_result.append([pair[0], pair[1]])
            new_cluster_index = list()
            for result in cluster_result:
                for i in result:
                    new_cluster_index.append(i)
            new_cluster_list = list()
            new_record_list = list()
            for i in range(len(cluster_result)):
                if not cluster_result[i]:
                    continue
                tmp = []
                for result in cluster_result[i]:
                    target = record_list[result-1]
                    tmp.extend(target)
                new_record_list.append(tmp)
            for i in range(1, len(record_list)+1):
                if i in new_cluster_index:
                    continue
                else:
                    new_record_list.append(record_list[i-1])
            for i in range(len(new_record_list)):
                cluster_index_list = np.array(new_record_list[i]) - 1
                if cluster_method == 'centroid':
                    new_cluster_list.append(np.mean(old_cluster_array[cluster_index_list], axis=0))
                else:
                    tmp_cluster_list = []
                    for j in cluster_index_list:
                        tmp_cluster_list.append(np.squeeze(old_cluster_array[j]))
                    new_cluster_list.append(tmp_cluster_list)
            cluster_list = new_cluster_list
            record_list = new_record_list
            print("the feature of cluster of {}-th step is\n {}\n".format(step, np.array(cluster_list)))
            print("the result of cluster of {}-th step is\n {}\n".format(step, record_list))
            step += 1

        print("the result of agglomerative hierarchical clustering is\n {}\n".format(record_list))
        return record_list

    # tutorial 5_4
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

    # tutorial 5_4
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

    # tutorial 5_18
    def hough_transform(self, image_region, theta):
        """
        """
        image_region_array = np.array(image_region)
        assert len(image_region_array.shape) == 2
        row = image_region_array.shape[0]-1
        column = image_region_array.shape[1]-1
        r = int(math.ceil(math.pow(row**2+column**2, 0.5)))
        r_list = [-i for i in range(1, r+1)] + [i for i in range(r+1)]
        accu_array = np.zeros((2*r+1, len(theta)))
        edge_pixel_index = np.where(image_region_array==1)
        for i in range(len(edge_pixel_index[0])):
            for tdx, t in enumerate(theta):
                y = edge_pixel_index[0][i]
                x = edge_pixel_index[1][i]
                t_r = math.radians(t)
                result = y * math.cos(t_r) - x * math.sin(t_r)
                if abs((2*math.ceil(result)-1) / 2 - result) < 1e-8:
                    result = int(math.ceil(result)-1)
                else:
                    result = int(round(result))
                accu_array[r-result][tdx] += 1
        return accu_array

    # tutorial 2_7
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

    # tutorial 2_11
    def compute_3d_point_2d_uv_coordinate(self, ori_coordinate, image_principal_point, magnification_factors, decimal=2):
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
        result_array = (1 / float(ori_coordinate[-1]) * np.dot(np.dot(array_1, array_2), array_3))
        return result_array
    
    def compute_3d_point_2d_xy_coordinate(self, ori_coordinate, f):
        """
        """
        assert len(ori_coordinate) == 3
        array_1 = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])
        ori_coordinate.append(1.0)
        array_2 = np.array(ori_coordinate).reshape(-1, 1)
        result_array = ((f / float(ori_coordinate[-2])) * np.dot(array_1, array_2))
        return result_array

    # tutorial 2_12
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

    # tutorial 3_1
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

    # tutorial 3_9
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
        

    # tutorial9
    def compute_similarity_using_normalised_cross_correlation(self, array_1, array_2):
        """
        """
        assert array_1.shape == array_2.shape
        result_1 = np.sum(array_1*array_2)
        result_2 = np.power(np.sum(np.power(array_1, 2)), 0.5)
        result_3 = np.power(np.sum(np.power(array_2, 2)), 0.5)
        return result_1 / (result_2 * result_3)
    
    def compute_similarity_using_cross_correlation(self, array_1, array_2):
        """
        """
        assert array_1.shape == array_2.shape
        return np.sum(array_1*array_2)
    
    def compute_similarity_using_correlation_coefficient(self, array_1, array_2):
        """
        """
        assert array_1.shape == array_2.shape
        array_1 = array_1 - np.mean(array_1)
        array_2 = array_2 - np.mean(array_2)
        result_1 = np.sum(array_1*array_2)
        result_2 = np.power(np.sum(np.power(array_1, 2)), 0.5)
        result_3 = np.power(np.sum(np.power(array_2, 2)), 0.5)
        return result_1 / (result_2 * result_3)

    def compute_min_dist(self, array_1, array_2):
        """
        """
        assert array_1.shape == array_2.shape
        correspond_index_1 = np.where(array_1==1)
        correspond_index_2 = np.where(array_2==1)

        result_list = []
        for i in range(len(correspond_index_1[0])):
            correspond_1 = np.array([correspond_index_1[0][i], correspond_index_1[1][i]])
            tmp_list = []
            for j in range(len(correspond_index_2[0])):
                correspond_2 = np.array([correspond_index_2[0][j], correspond_index_2[1][j]])
                tmp_list.append(self.compute_eucli_distance(correspond_1, correspond_2))
            result_list.append(min(tmp_list))
        return np.mean(np.array(result_list))

    # tutorial 9_1, 9_4
    def find_object_location(self, template, image, method):
        """
        """
        template = np.array(template)
        image = np.array(image)
        assert template.shape[0] == template.shape[1]
        interval = (template.shape[0] - 1) // 2
        start = [interval, interval]
        end =  [image.shape[0] - 1 - start[0], image.shape[1] - 1 - start[1]]
        if method == 'normalised_cross_correlation':
            result_array = np.zeros(image.shape)
            for i in range(start[0], end[0]+1):
                for j in range(start[1], end[1]+1):
                    array_1 = template
                    array_2 = image[i-interval:i+interval+1, j-interval:j+interval+1]
                    result_array[i][j] = self.compute_similarity_using_normalised_cross_correlation(array_1, array_2)
            print("the template result of pixel in image is\n {}\n".format(result_array))
            result = np.unravel_index(result_array.argmax(), result_array.shape)

        elif method == 'cross_correlation':
            result_array = np.zeros(image.shape)
            for i in range(start[0], end[0]+1):
                for j in range(start[1], end[1]+1):
                    array_1 = template
                    array_2 = image[i-interval:i+interval+1, j-interval:j+interval+1]
                    result_array[i][j] = self.compute_similarity_using_cross_correlation(array_1, array_2)
            print("the template result of pixel in image is\n {}\n".format(result_array))
            result = np.unravel_index(result_array.argmax(), result_array.shape)

        elif method == 'correlation_coefficient':
            result_array = np.zeros(image.shape)
            for i in range(start[0], end[0]+1):
                for j in range(start[1], end[1]+1):
                    array_1 = template
                    array_2 = image[i-interval:i+interval+1, j-interval:j+interval+1]
                    result_array[i][j] = self.compute_similarity_using_correlation_coefficient(array_1, array_2)
            print("the template result of pixel in image is\n {}\n".format(result_array))
            result = np.unravel_index(result_array.argmax(), result_array.shape)

        elif method == 'sum_of_absolute_differences':
            result_array = np.full(image.shape, fill_value=float('inf'))
            for i in range(start[0], end[0]+1):
                for j in range(start[1], end[1]+1):
                    array_1 = template
                    array_2 = image[i-interval:i+interval+1, j-interval:j+interval+1]
                    result_array[i][j] = self.compute_SAD_diff(array_1, array_2)
            print("the template result of pixel in image is\n {}\n".format(result_array))
            result = np.unravel_index(result_array.argmin(), result_array.shape)

        elif method == 'minimum_distance':
            result_array = np.full(image.shape, fill_value=float('inf'))
            for i in range(start[0], end[0]+1):
                for j in range(start[1], end[1]+1):
                    array_1 = template
                    array_2 = image[i-interval:i+interval+1, j-interval:j+interval+1]
                    result_array[i][j] = self.compute_min_dist(array_1, array_2)
            print("the template result of pixel in image is\n {}\n".format(result_array))
            result = np.unravel_index(result_array.argmin(), result_array.shape)
    
        elif method == 'eucli_distance':
            result_array = np.full(image.shape, fill_value=float('inf'))
            for i in range(start[0], end[0]+1):
                for j in range(start[1], end[1]+1):
                    array_1 = template
                    array_2 = image[i-interval:i+interval+1, j-interval:j+interval+1]
                    result_array[i][j] = self.compute_eucli_distance(array_1, array_2)
            print("the template result of pixel in image is\n {}\n".format(result_array))
            result = np.unravel_index(result_array.argmin(), result_array.shape)
        
        return [result[1]+1, result[0]+1]

    # tutorial 9_3
    def best_template_match(self, template_list, image, method, decimal):
        """
        """
        if method == 'all':
            self.best_template_match(template_list, image, method='cross_correlation', decimal=decimal)
            self.best_template_match(template_list, image, method='normalised_cross_correlation', decimal=decimal)
            self.best_template_match(template_list, image, method='correlation_coefficient', decimal=decimal)
            self.best_template_match(template_list, image, method='SAD', decimal=decimal)
            return

        image = np.array(image)
        if method == 'cross_correlation':
            result_list = []
            for template in template_list:
                result_list.append(round(self.compute_similarity_using_cross_correlation(np.array(template), image), decimal))
            print("the cross correlation result is\n {}\n".format(result_list))
            max_value = max(result_list)
            final_result_list = []
            for idx, val in enumerate(result_list):
                if val == max_value:
                    final_result_list.append(idx+1)
            return final_result_list
        elif method == 'normalised_cross_correlation':
            result_list = []
            for template in template_list:
                result_list.append(round(self.compute_similarity_using_normalised_cross_correlation(np.array(template), image), decimal))
            print("the normalised cross correlation result is\n {}\n".format(result_list))
            max_value = max(result_list)
            final_result_list = []
            for idx, val in enumerate(result_list):
                if val == max_value:
                    final_result_list.append(idx+1)
            return final_result_list
        elif method == 'correlation_coefficient':
            result_list = []
            for template in template_list:
                result_list.append(round(self.compute_similarity_using_correlation_coefficient(np.array(template), image), decimal))
            print("the correlation coefficient result is\n {}\n".format(result_list))
            max_value = max(result_list)
            final_result_list = []
            for idx, val in enumerate(result_list):
                if val == max_value:
                    final_result_list.append(idx+1)
            return final_result_list
        elif method == 'SAD':
            result_list = []
            for template in template_list:
                result_list.append(round(self.compute_SAD_diff(np.array(template), image), decimal))
            print("the SAD result is\n {}\n".format(result_list))
            min_value = min(result_list)
            final_result_list = []
            for idx, val in enumerate(result_list):
                if val == min_value:
                    final_result_list.append(idx+1)
            return final_result_list

    # tutorial 9_10
    def compute_cross_ratio(self, p1, p2, p3, p4, method='3d', center_coordinate=None, magnification_factors=None, decimal=2):
        """
        """
        if method == '2d':
            assert center_coordinate
            assert magnification_factors
            p1 = self.compute_3d_point_2d_uv_coordinate(ori_coordinate=p1, image_principal_point=center_coordinate, magnification_factors=magnification_factors, decimal=decimal)
            p2 = self.compute_3d_point_2d_uv_coordinate(ori_coordinate=p2, image_principal_point=center_coordinate, magnification_factors=magnification_factors, decimal=decimal)
            p3 = self.compute_3d_point_2d_uv_coordinate(ori_coordinate=p3, image_principal_point=center_coordinate, magnification_factors=magnification_factors, decimal=decimal)
            p4 = self.compute_3d_point_2d_uv_coordinate(ori_coordinate=p4, image_principal_point=center_coordinate, magnification_factors=magnification_factors, decimal=decimal)
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)
        dist_1_3 = round(self.compute_eucli_distance(p1, p3), decimal)
        dist_2_4 = round(self.compute_eucli_distance(p2, p4), decimal)
        dist_1_4 = round(self.compute_eucli_distance(p1, p4), decimal)
        dist_2_3 = round(self.compute_eucli_distance(p2, p3), decimal)
        print(dist_1_3, dist_2_4, dist_1_4, dist_2_3)
        cross_ratio = round((dist_1_3 * dist_2_4) / (dist_1_4 * dist_2_3))
        return cross_ratio
    