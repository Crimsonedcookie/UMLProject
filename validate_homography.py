import cv2
from aruco_identification import detect_markers_by_id
import numpy as np
import statistics as stats
from collect_labelled_points import *
from utility_funcs import (make_detector,resolve_dictionary)
from pipeline_config import PipelineConfig
from sklearn.metrics import mean_squared_error
from math import sqrt

source_path="/Users/nathanramani/PycharmProjects/UML/SineWaveTest1_corrected/sine_test1_frame_0_corrected.png"
source_image=cv2.imread(str(source_path), cv2.IMREAD_COLOR)


config = PipelineConfig()
dictionary = resolve_dictionary(config.dictionary_name)
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


x_rmse=[]
y_rmse=[]


def collect_point_coordinates(image):
    points,_=collect_labeled_points(image=image,detector=detector,dictionary=dictionary,config=config)
    x=[]
    y=[]
    for point in points.values():
        x.append(point[0])
        y.append(point[1])
    return x,y


def rmse_compare(actual_data,source_data):
    mse=mean_squared_error(actual_data,source_data)
    rmse=sqrt(mse)
    return rmse


source_x,source_y=collect_point_coordinates(source_image)


for i in range(0,605,1):
    validation_path = "/Users/nathanramani/PycharmProjects/UML/SineWaveTest1_corrected/sine_test1_frame_"+str(i)+"_corrected.png"
    validation_image = cv2.imread(str(validation_path), cv2.IMREAD_COLOR)

    if validation_image is None:
        print(f"Couldn't load frame {i}")
        continue

    current_x,current_y= collect_point_coordinates(validation_image)

    x_current_rmse=rmse_compare(current_x,source_x)
    y_current_rmse=rmse_compare(current_y,source_y)

    x_rmse.append(x_current_rmse)
    y_rmse.append(y_current_rmse)

#Interpreting RMSE
print("Average RMSE: x="+str(np.mean(x_rmse))+" y="+str(np.mean(y_rmse)))
print("Max RMSE: x="+str(np.max(x_rmse))+" y="+str(np.max(y_rmse)))
