from flask import Flask
from flask import request
from flask import jsonify
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import math
import random

app = Flask(__name__)

number_of_features = 31   # input size
number_of_features = 28   # input size
number_of_hidden   = 32   # size of hidden layer
number_of_gestures = 12   # output size
sequence_length    = 20   # refers to the amount of timeframes to check
folder_name = ["none", "pinch_in", "pinch_out", "swipe_up", "swipe_down",
               "swipe_left", "swipe_right", "grab2fist", "fist2grab", "peace",
               "2fingers", "pointing"]

# 0 - No gesture
# 1 - Pinch zoom in
# 2 - Pinch zoom out
# 3 - Swipe up
# 4 - Swipe down
# 5 - Swipe left
# 6 - Swipe right
# 7 - grab to fist
# 8 - fist to grab
# 9 - peace sign to exit
# 10 - 2 fingers for "selecting" button
# 11 - 1 finger to act as mouse

class JarvisLSTM(nn.Module):
    def __init__(self, hidden_dim, input_size, gesture_size, seq_len):
        super(JarvisLSTM, self).__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_len
        self.gesture_size = gesture_size
       
        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=1 , batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim * seq_len, gesture_size)

    def forward(self, seq_batch_input):

        batch_size = list(seq_batch_input.size())[1]

        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)

        # run the lstm model
        lstm_out, (ht, ct) = self.lstm(seq_batch_input, hidden)

        #print(lstm_out.shape)
        #lstm_out = lstm_out[-1, :, :]

        # lstm_out = (seq_len, batch, hidden_size)
        # convert lstm_out to (batch_size, -1) to merge
        lstm_out = lstm_out.contiguous().view(batch_size, -1)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out)

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        #gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        #return gesture_probabilities
        return gesture_out

lstm_model = JarvisLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)
lstm_model.load_state_dict(torch.load("atan.model"))
lstm_model.load_state_dict(torch.load("state_dict_model_wrist_features30epoch.pt"))

lstm_model.eval()

feature_order = ["pitch", "roll", "yaw", "thumb2index", "thumb2middle", "thumb2ring", "thumb2pinky", "thumbFromStart", "indexFromStart",
                 "middleFromStart", "ringFromStart", "pinkyFromStart", "thumbLength", "indexLength", "middleLength",
                 "ringLength", "pinkyLength", "index_omega", "middle_omega", "ring_omega", "pinky_omega",
                 "index_beta", "middle_beta", "ring_beta", "pinky_beta", "thumb_index_gamma", "index_middle_gamma",
                 "middle_ring_gamma", "ring_pinky_gamma", "wrist_phi", "wrist_theta"]

qml_data = [0, 0, 0, 0]

@app.route("/get-gesture/", methods=['POST'])
def get_gesture():
    ret = {"gesture" : qml_data[0],
           "x" : qml_data[1],
           "y" : qml_data[2],
           "z" : qml_data[3]}
    return jsonify(ret)

# Flask Server stuff
@app.route("/determine-gesture/", methods=['POST'])
def determine_gesture():
    # json_data is a dictionary indexed by timestep
    json_data = request.get_json(force=True)
    json_data = json.loads(json_data)

    # first turn the data into correct order in the list
    smol_array = [[]] * len(feature_order)
    format_data = ([smol_array] * sequence_length)
    for i in range(0, sequence_length):
        for j in range(0, len(feature_order)):
            format_data[i][j] = json_data[str(i)][feature_order[j]]

    # create tensor
    eval_tensor = torch.FloatTensor(format_data)

    # send the information through the model
    result = lstm_model(eval_tensor.view(sequence_length, 1, number_of_features))

    # manipulate model to get item with highest
    # probability
    result = torch.argmax(result)

    qml_data[0] = result.item()
    qml_data[1] = json_data["x"]
    qml_data[2] = json_data["y"]
    qml_data[3] = json_data["z"]

    # send that information back properly
    result = {"gesture" : result.item()}
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
