# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:50:53 2017

@author: aumale
"""
import zipfile as zipf
import scipy
import numpy as np

class LoadUSPS:
    def __init__(self,zipLocation):
        self.zipLocation = zipLocation
    
    def load(self,one_hot=False, flatten=True):
        zip = zipf.ZipFile(self.zipLocation,'r')
        usps_img = []
        usps_lbl = []
		
        filelist = zip.namelist()
        filelist.sort()
        for file in filelist:
			
            if 'Numerals' in file and '.png' in file:
                # extrach image as grayscale, compress in 28*28, append in usps_img array
                with zip.open(file, 'r') as img:
                    img_arr = scipy.misc.imread(img,mode='L')
                    
                    #make 0 = black and 1 = white and scale rest in between
                    img_arr = abs((img_arr-255.0)*1/255).round(3)
                    
                    #crop around digit
                    img_arr = self._cropping(img_arr)
                    
                    #pad around digit to add blank space
                    img_arr = self._padding(img_arr, 10, 10, 10, 10)
                    
                    #resize the image to 28*28
                    img_arr= scipy.misc.imresize(img_arr, (28, 28))
                    
                    #flatten image to give flat vector instead of 28*28 matrix
                    if flatten == True:
                        img_arr = img_arr.flatten()
                    
                    usps_img.append(img_arr)
                    if one_hot == True :
                        one_hot_arr = [0]*10
					
                    for i in range(10):
                        if '/'+str(i)+'/' in file:
                            if one_hot == True:
                                one_hot_arr[i] = 1
                                usps_lbl.append(one_hot_arr)
                            else:
                                usps_lbl.append(i)
                            break
                        
        usps_img = np.array(usps_img)        
        usps_lbl = np.array(usps_lbl,dtype='uint8')
        return usps_img, usps_lbl
    
    def _padding(self, img, pad_l, pad_t, pad_r, pad_b):
        height, width = img.shape
        #Adding padding to the left side.
        pad_left = np.zeros((height, pad_l), dtype = np.int)
        img = np.concatenate((pad_left, img), axis = 1)
        
        #Adding padding to the top.
        pad_up = np.zeros((pad_t, pad_l + width))
        img = np.concatenate((pad_up, img), axis = 0)
        
        #Adding padding to the right.
        pad_right = np.zeros((height + pad_t, pad_r))
        img = np.concatenate((img, pad_right), axis = 1)

        #Adding padding to the bottom
        pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
        img = np.concatenate((img, pad_bottom), axis = 0)
        
        return img

    def _cropping(self, img):
        col_sum = np.where(np.sum(img, axis = 0)>0)
        row_sum = np.where(np.sum(img, axis = 1)>0)
        y1, y2 = row_sum[0][0], row_sum[0][-1]
        x1, x2 = col_sum[0][0], col_sum[0][-1]
        return img[y1:y2, x1:x2]