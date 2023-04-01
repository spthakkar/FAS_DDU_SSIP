# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:38:22 2022

@author: punja
"""


#MTCNN on MAFA

#IMPORT NECESSARY LIBRARIES
import os 
from os import listdir
import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import numpy as np
c1 = cv2.getTickCount()   #START TIMER
folder= 'C:\\Users\\punja\\Documents\\SEM 8\\MAFA\\test-images\\t\\multi face' #FOLDER LOCATION WHERE IMAGES ARE STORED
indexN=0

def face_extractor(img):
    detector = MTCNN()  #LOAD MTCNN MODEL
    faces = detector.detect_faces(img)  #GIVE IMAGE AS AN INPUT TO MODEL
    print('faces:',faces)
    
    if faces == ():
        return None

    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x,y), (x1,y1), (0,0,0), 8)   #DRAW RECTANGULAR BOUNDING BOX
              
    return img
#cv2.imwrite('image0-pred.jpg', image)
    plt.imshow(img)

for images in os.listdir(folder):
    count=0
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        print(images)
        #frame=cv2.imread(folder+images)
        
        f = os.path.join(folder, images)
        x=os.path.splitext(os.path.basename(f))[0]
                                                                    
        frame=cv2.imread(f)
        
        
       # print(frame)
        file_name_path1 = 'C:\\Users\\punja\\Documents\\SEM 8\\MAFA\\test-images\\t\\multi_face_out\\MT_'+str(indexN)+images+'.jpg'   #path to store image
        image_90_clk = face_extractor(frame)
        
        
        #cv2.imwrite(file_name_path1,image_90_clk)
        
        indexN=indexN+1
        count=count+1    
          
c2 = cv2.getTickCount()  #STOP TIMER
time_taken = (c2 - c1)/ cv2.getTickFrequency() 
print(f'The time taken for execution is {time_taken}') 