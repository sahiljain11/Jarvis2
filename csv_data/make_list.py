import os
import numpy as np
import pandas as pd
from progress.bar import IncrementalBar

#folder = "2fingers"

basedir = os.path.abspath(os.path.dirname(__file__))

data_file = os.path.join(basedir, "2fingers/2fingers0.csv")

f = open(data_file, "r")
file_data = f.readlines()

first_row = file_data[0].split(",")

#cols = [2, 3, 4, 5, 6, 7]
distals = []

for i in range(0, len(first_row)):
    if "distal_end" in first_row[i]:
        distals.append(i)


#print(len(cols))
#print(cols)

finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

# add more relevant features to the csvs
folder_name = ["none", "pinch_in", "pinch_out", "swipe_up", "swipe_down",
               "swipe_left", "swipe_right", "grab2fist", "fist2grab", "peace",
               "2fingers", "pointing"]

def find_distance_between_vec(df, col1, col2):
    return ((df[col1 + "x"] - df[col2 + "x"])**2 + (df[col1 + "y"] - df[col2 + "y"])**2 + (df[col1 + "z"] - df[col2 + "z"])**2)**0.5


for i in range(9, len(folder_name)):
    name = folder_name[i]
    file_direc = os.path.join(basedir, name + "/")
    files = os.listdir(file_direc)

    with IncrementalBar("Creating " + str(name) + "...", max=len(files)) as increment_bar:
        for iter_file in files:
            # load data frame
            file_loc = file_direc + iter_file
            df = pd.read_csv(file_loc)

            df = df.rename(columns={"Unnamed: 197" : "thumb2index"})

            # calculate distances between each of the fingers to the thumb
            for i in range(1, len(finger_names)):
                df["thumb2" + finger_names[i]] = find_distance_between_vec(df, "thumb_distal_end_", finger_names[i] + "_distal_end_")

            # change in each finger location
            for i in range(0, len(finger_names)):
                df[finger_names[i] + "FromStart"] = df[finger_names[i] + "_distal_end_x"] - df.loc[0, finger_names[i] + "_distal_end_x"]

            for i in range(0, len(finger_names)):
                df[finger_names[i] + "Length"] = find_distance_between_vec(df, "hand_position_", finger_names[i] + "_distal_end_")
                    
            # save the csv
            df.to_csv(file_loc, index=False, header=True)
            increment_bar.next()
