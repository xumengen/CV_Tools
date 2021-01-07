# CV_Tools

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
This repo contains the functions which appear in the turorails.
  - Let's gogogo
  
    - [Main Features](#main-features)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Todos](#todos)


### Main Features

| Function Name  | Tutorial | Input Parameter | Output Parameter | annotation |
| :--------------: | :--------: | :---------------: | :----------------: | :----------: |
| compute_thin_lens_equation | 2.7  | f, z1, z2 | f or z1 or z2| compute one parameter using thin lens equation |
|  compute_3d_point_2d_uv_coordinate  | 2.11 | ori_coordinate, image_principal_point, magnification_factors, decimal | 2d point |compute the 3d point in the uv image coordinate system |
|  compute_3d_point_2d_xy_coordinate  | 2.11 | ori_coordinate, f, decimal | 2d point |compute the 3d point in the xy image coordinate system |
|  convert_rbg_to_gray | 2.12 | ori_image, bit | gray image | convert rgb image to gray image |
| convolution | 3.1 | mask, I, method | convolution result | convolve image |
| compute_pixel_val_using_gaussian | 3.9 | array_size, standard_deviation, decimal | each pixel value | compute each pixel using gaussian formula |
| dilation | 5.4 | input_array, mode | dilation result | dialate image |
| erosion | 5.4 | input_array, mode | erosion result | erode image |
| region_growing | 5.6 | feature_vector_array, method, thres, mode, start | region result | segment using region growing |
| region_merge | 5.8 | feature_vector_array, method, thres, mode, start, result_array | region result | segment using region merge |
| region_split_and_merge | 5.10 | feature_vector_array, method, thres, mode, start | region result | segment using region split and merge |
| k_means | 5.12 | feature_vector_array, k, ori_feature_vetor_array, method | region result | segment using k means |
| agglomerative_hierarchical_clusteringv2 | 5.15 | feature_vector_array, k, method, cluster_method | region result |segment using hierarchical clustering |
| hough_transform | 5.18 | image_region, theta | accumulator array | perform hough transform on the image |
| compute_similarity_between_point_and_image | 6.4 | coordinate, left_image, right_image, k | similarity array | compute the similarity of one pixel of left image with the right image |
| compute_dist_between_two_feature_vector_array | 6.5 | array_1, array_2, method | dist array | compute dist between two arrays |
| RANSAC | 6.8 | left_point, right_point, thres, trials, decimal | best estimation of the model | find the true correspondence between the two images |
| compute_harris_corner_detector | 6.9 | Ix, Iy, k, length | R array | computer harris corner detector |
| compute_z_from_two_coplanar_camera | 7.1 |  f, B, left_coordinate, right_coordinate, pixel_size, coplanar, decimal | z value | compute z value from two coplanar camera |
| compute_distance_between_point_and_horopter | 7.5 | baseline_length, angle_z_baseline, a_l, a_r | distance | compute distance between point and horoper |
| compute_depth_of_scence_point | 8.4 | frame_1_point, frame_2_point, velocity, move_method, pixel_size, focal_length, center_coordinate | depth value | compute depth value |
| compute_segment_moving_object_from_background | 8.5 | pixel_patches, thres, beta, method | segment result | segment the moving object from background |
| find_object_location | 9.1 9.4 | template, image, method | similarity array | find object location in image |
| best_template_match | 9.3 | template_list, image, method, decimal | the the template which matches the image best | find the best match template |
| compute_cross_ratio | 9.10 | p1, p2, p3, p4, method, center_coordinate, magnification_factors, decimal | cross ratio | compute cross ratio |
| compute_object_class | 10.4 | class_list, feature_vector_list, object_feature_vector, k | class label | compute the class of the object |


### Installation

```
git clone --recursive https://github.com/xumengen/CV_Tools.git
cd CV_Tools
enjoy it!
```

### Usage

```
step_1: choose the function you want to ues in "main.py"
step_2: run "python main.py"
```


### Todos

 - 

**Free Software, Yeah!**
