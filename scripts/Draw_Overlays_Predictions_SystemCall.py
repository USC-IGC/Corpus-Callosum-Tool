# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:06:25 2021

@author: abhi8
"""
import cv2
import os
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--it")
parser.add_argument("--path")
args=parser.parse_args()
path_img=args.path+"/Images_PNG/"
path_mask=args.path+"/Labels_PNG/"
path_draw=args.path+"/Overlays_PNG/"

it=args.it[1:-1]
#print(it,path_img,path_mask,path_draw)
img=cv2.imread(path_img+it)
mask=cv2.imread(path_mask+it,cv2.IMREAD_GRAYSCALE)
ret, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
cv2.imwrite(path_draw+it, img)
            
