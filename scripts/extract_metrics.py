#!/usr/bin/env python
import sys
from subprocess import check_output
import argparse
import os
from glob import glob
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize, thin, binary_dilation
from skimage.measure import regionprops, label, find_contours
import cv2
from skimage.transform import rotate
from scipy.optimize import leastsq
import pandas as pd
import json

import warnings
warnings.simplefilter("ignore")


def extract_cc(image):
    hdr = nib.load(image)
    img_data = hdr.get_fdata()
    midslice = img_data[89,:,:]
    cc = np.rollaxis(midslice, 1)
    return cc


def thickness_calc(corcal):
    try:
        contours = find_contours(corcal, 0)
        cl = 0
        # if len(contours) == 0:
        #     return np.nan, np.nan, np.nan, np.nan
        for i, c in enumerate(contours):
            if len(c) > cl:
                print()
                cl = len(c)
                outline = contours[i]

        dic = {}
        for i in range(0, len(outline)):
            if outline[i][1] in dic.keys():
                dic[outline[i][1]].append(outline[i][0])
            else:
                dic[outline[i][1]] = [outline[i][0]]

        points = []
        for k,v in dic.items():
            if len(v)==2:
                points.append(k)

        thickness=[]
        for point in points:
            thickness.append(np.abs(dic[point][0]-dic[point][1]))

        thickness = np.array(thickness)
        return np.nanmean(thickness), np.nanstd(thickness), np.nanmax(thickness), np.nanmin(thickness)
    except:
        return np.nan, np.nan, np.nan, np.nan


def eucldist_calc(corcal):
    contours = find_contours(corcal, 0)[0]
    
    contours_sorted = sorted(contours, key=lambda x: x[0])
    cc_len = np.sqrt(np.sum((np.square(contours_sorted[0][0]-contours_sorted[-1][0]), np.square(contours_sorted[0][1]-contours_sorted[-1][1]))))
    return cc_len
    
    
def perimeter_calc(corcal):
    gray = np.uint8(corcal * 255)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    #fixed// https://stackoverflow.com/questions/64345584/how-to-properly-use-cv2-findcontours-on-opencv-version-4-4-0
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_len = cv2.arcLength(contours[0], True)
    return cnt_len


def curve_calc(path_array):
    x = np.where(path_array)[0]
    y = np.where(path_array)[1]
    
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x_m = np.nanmean(x)
    y_m = np.nanmean(y)
    center_estimate = np.array([x_m, y_m])
    try:
        center_2, ier = leastsq(f_2, center_estimate)

        # xc_2, yc_2 = center_2
        Ri_2       = calc_R(*center_2)
        R_2        = Ri_2.mean()
        # residu_2   = sum((Ri_2 - R_2)**2)
        return 1/R_2
    except:
        return np.nan


def pointwise_curve_calc(path_array):
    try:
        dx_dt = np.gradient(np.where(path_array)[0])
        dy_dt = np.gradient(np.where(path_array)[1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs((dx_dt*d2y_dt2 - dy_dt*d2x_dt2)/((dx_dt**2 + dy_dt**2)**(3/2)))

        return np.nanmean(curvature), np.nanstd(curvature), np.nanmax(curvature), np.nanmin(curvature)
    except:
        return np.nan, np.nan, np.nan, np.nan


def remove_skeleton_branches(skeleton):
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", help="Mask image")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()

    ### Preprocessing
    image  = extract_cc(args.mask)
    thr001 = (image > 0.001).astype(int)
    labeled = label(thr001)
    
    area = 0
    cc_label = 0
    orientation = None
    for i, prop in enumerate(regionprops(labeled)):
        if prop.area > area:
            area = prop.area
            cc_label = i+1
            orientation = prop.orientation

    cc = (labeled == cc_label).astype(int)
    skeleton = thin(cc)
    rotate_by = np.sign(orientation)*(90 - np.abs(orientation) * 180 / np.pi)
    rotated = rotate(cc, angle=rotate_by, order=0, preserve_range=True)

    ### Subregion Segmentation
    y_min = np.min(np.where(rotated)[1])
    y_max = np.max(np.where(rotated)[1])
    straight_length = y_max - y_min

    # Witelson segmentation
    witelson5 = np.ones_like(rotated)
    witelson5[:,(y_max-int(1/2.*straight_length)):(y_max-int(1/3.*straight_length))] = 2
    witelson5[:,(y_max-int(2/3.*straight_length)):(y_max-int(1/2.*straight_length))] = 3
    witelson5[:,(y_max-int(4/5.*straight_length)):(y_max-int(2/3.*straight_length))] = 4
    witelson5[:,:y_max-int(4/5.*straight_length)] = 5

    rotated_witelson5 = rotate(witelson5, angle=-rotate_by, order=0, preserve_range=True)
    cc_witelson5 = cc * rotated_witelson5
    cc_witelson5_copy = cc * rotated_witelson5

    # JHU segmentation
    jhu3 = np.ones_like(rotated)
    jhu3[:,(y_max-int(5/6.*straight_length)):(y_max-int(1/6.*straight_length))] = 2
    jhu3[:,:y_max-int(5/6.*straight_length)] = 3

    rotated_jhu3 = rotate(jhu3, angle=-rotate_by, order=0, preserve_range=True)
    cc_jhu3 = cc * rotated_jhu3
    cc_jhu3_copy = cc * rotated_jhu3

    # Fix Genu Tips
    for atlas, num_regions in zip([cc_witelson5_copy, cc_jhu3_copy], [5, 3]):
        for i in range(2, num_regions+1):
            subregion = (atlas == i).astype(int)
            sublabels, num_labels = label(subregion, return_num=True)
            if num_labels > 1:
                to_convert = 0
                area_to_convert = 0
                for n in range(1, num_labels+1):
                    if (sublabels == n).astype(int).sum() > area_to_convert:
                        to_convert = n
                atlas[atlas == to_convert] = 1

    skel_witelson5 = skeleton * rotated_witelson5
    skel_jhu3 = skeleton * rotated_jhu3

    ### Metric Extraction

    mdict = {}
    # Whole region metrics
    mdict["Total_Area"] = np.sum(cc)   # Total area
    mdict["Total_Curve"] = curve_calc(skeleton)   #Total curvature
    mdict["Total_MeanCurve"], mdict["Total_StdCurve"], mdict["Total_MaxCurve"], mdict["Total_MinCurve"] = pointwise_curve_calc(skeleton)  
    mdict["Total_MeanThickness"], mdict["Total_StdThickness"], mdict["Total_MaxThickness"], mdict["Total_MinThickness"] = thickness_calc(rotated)    # Thickness  
    mdict["Total_Perimeter"] = perimeter_calc(rotated)   # Perimeter
    mdict["Total_EuclideanDist"] = eucldist_calc(rotated)   # Maximum euclidean distance
    mdict["Total_MedialCurveLength"] = np.sum(skeleton)   # Medial curve length
    mdict["Ratio_MedialCurve_MaxEuclideanDist"] = np.sum(skeleton)/eucldist_calc(rotated)   # Ratio of medial curve to max euclidean dist to measure hump
   
    # Whitelson5_Genu metrics
    mdict["Witelson5_Genu_Area"] = np.sum((cc_witelson5 == 1).astype(int))   #Area
    mdict["Witelson5_Genu_Curve"] = curve_calc((skel_witelson5 == 1).astype(int))    # Overall curvature
    mdict["Witelson5_Genu_MeanCurve"], mdict["Witelson5_Genu_StdCurve"], mdict["Witelson5_Genu_MaxCurve"], mdict["Witelson5_Genu_MinCurve"] = pointwise_curve_calc((skel_witelson5 == 1).astype(int))   # Pointwise Curvature
    mdict["Witelson5_Genu_MeanThickness"], mdict["Witelson5_Genu_StdThickness"], mdict["Witelson5_Genu_MaxThickness"], mdict["Witelson5_Genu_MinThickness"] = thickness_calc((cc_witelson5 ==1).astype(int)) 

    # Witelson5_AnteriorBody metrics
    mdict["Witelson5_AnteriorBody_Area"] = np.sum((cc_witelson5 == 2).astype(int))
    mdict["Witelson5_AnteriorBody_Curve"] = curve_calc((skel_witelson5 == 2).astype(int))
    mdict["Witelson5_AnteriorBody_MeanCurve"], mdict["Witelson5_AnteriorBody_StdCurve"], mdict["Witelson5_AnteriorBody_MaxCurve"], mdict["Witelson5_AnteriorBody_MinCurve"] = pointwise_curve_calc((skel_witelson5 == 2).astype(int))
    mdict["Witelson5_AnteriorBody_MeanThickness"], mdict["Witelson5_AnteriorBody_StdThickness"], mdict["Witelson5_AnteriorBody_MaxThickness"], mdict["Witelson5_AnteriorBody_MinThickness"] = thickness_calc((cc_witelson5 ==2).astype(int)) 

    # Witelson5_PosteriorBody metrics
    mdict["Witelson5_PosteriorBody_Area"] = np.sum((cc_witelson5 == 3).astype(int))
    mdict["Witelson5_PosteriorBody_Curve"] = curve_calc((skel_witelson5 == 3).astype(int))
    mdict["Witelson5_PosteriorBody_MeanCurve"], mdict["Witelson5_PosteriorBody_StdCurve"], mdict["Witelson5_PosteriorBody_MaxCurve"], mdict["Witelson5_PosteriorBody_MinCurve"] = pointwise_curve_calc((skel_witelson5 == 3).astype(int))
    mdict["Witelson5_PosteriorBody_MeanThickness"], mdict["Witelson5_PosteriorBody_StdThickness"], mdict["Witelson5_PosteriorBody_MaxThickness"], mdict["Witelson5_PosteriorBody_MinThickness"] = thickness_calc((cc_witelson5 == 3).astype(int)) 

    # Witelson5_Isthmus metrics 
    mdict["Witelson5_Isthmus_Area"] = np.sum((cc_witelson5 == 4).astype(int))
    mdict["Witelson5_Isthmus_Curve"] = curve_calc((skel_witelson5 == 4).astype(int))
    mdict["Witelson5_Isthmus_MeanCurve"], mdict["Witelson5_Isthmus_StdCurve"], mdict["Witelson5_Isthmus_MaxCurve"], mdict["Witelson5_Isthmus_MinCurve"] = pointwise_curve_calc((skel_witelson5 == 4).astype(int))
    mdict["Witelson5_Isthmus_MeanThickness"], mdict["Witelson5_Isthmus_StdThickness"], mdict["Witelson5_Isthmus_MaxThickness"], mdict["Witelson5_Isthmus_MinThickness"] = thickness_calc((cc_witelson5 == 4).astype(int)) 

    # Witelson5_Splenium metrics
    mdict["Witelson5_Splenium_Area"] = np.sum((cc_witelson5 == 5).astype(int))
    mdict["Witelson5_Splenium_Curve"] = curve_calc((skel_witelson5 == 5).astype(int))
    mdict["Witelson5_Splenium_MeanCurve"], mdict["Witelson5_Splenium_StdCurve"], mdict["Witelson5_Splenium_MaxCurve"], mdict["Witelson5_Splenium_MinCurve"] = pointwise_curve_calc((skel_witelson5 == 5).astype(int))
    mdict["Witelson5_Splenium_MeanThickness"], mdict["Witelson5_Splenium_StdThickness"], mdict["Witelson5_Splenium_MaxThickness"], mdict["Witelson5_Splenium_MinThickness"] = thickness_calc((cc_witelson5 == 5).astype(int)) 

    # JHU3_Genu metrics
    mdict["JHU3_Genu_Area"] = np.sum((cc_jhu3 == 1).astype(int))
    mdict["JHU3_Genu_Curve"] = curve_calc((skel_jhu3 == 1).astype(int))
    mdict["JHU3_Genu_MeanCurve"], mdict["JHU3_Genu_StdCurve"], mdict["JHU3_Genu_MaxCurve"], mdict["JHU3_Genu_MinCurve"] = pointwise_curve_calc((skel_jhu3 == 1).astype(int))
    mdict["JHU3_Genu_MeanThickness"], mdict["JHU3_Genu_StdThickness"], mdict["JHU3_Genu_MaxThickness"], mdict["JHU3_Genu_MinThickness"] = thickness_calc((cc_jhu3 == 1).astype(int)) 

    # JHU3_Body metrics
    mdict["JHU3_Body_Area"] = np.sum((cc_jhu3 == 2).astype(int))
    mdict["JHU3_Body_Curve"] = curve_calc((skel_jhu3 == 2).astype(int))
    mdict["JHU3_Body_MeanCurve"], mdict["JHU3_Body_StdCurve"], mdict["JHU3_Body_MaxCurve"], mdict["JHU3_Body_MinCurve"] = pointwise_curve_calc((skel_jhu3 == 2).astype(int))
    mdict["JHU3_Body_MeanThickness"], mdict["JHU3_Body_StdThickness"], mdict["JHU3_Body_MaxThickness"], mdict["JHU3_Body_MinThickness"] = thickness_calc((cc_jhu3 == 2).astype(int)) 

    # JHU3_Splenium metrics
    mdict["JHU3_Splenium_Area"] = np.sum((cc_jhu3 == 3).astype(int))
    mdict["JHU3_Splenium_Curve"] = curve_calc((skel_jhu3 == 3).astype(int))
    mdict["JHU3_Splenium_MeanCurve"],mdict["JHU3_Splenium_StdCurve"], mdict["JHU3_Splenium_MaxCurve"], mdict["JHU3_Splenium_MinCurve"] = pointwise_curve_calc((skel_jhu3 == 3).astype(int))
    mdict["JHU3_Splenium_MeanThickness"], mdict["JHU3_Splenium_StdThickness"], mdict["JHU3_Splenium_MaxThickness"], mdict["JHU3_Splenium_MinThickness"] = thickness_calc((cc_jhu3 == 3).astype(int)) 

 
    
    # Write Output Spreadsheet
    metrics = pd.DataFrame.from_dict(mdict, orient='index')
    # metrics = metrics[sorted(metrics.columns)]
    metrics.to_csv(args.output, index_label="Measures", header=['Value'])



    
