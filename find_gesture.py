from flask import Flask
from flask import request
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

number_of_features = 28  # input size
number_of_hidden   = 32   # size of hidden layer
number_of_gestures = 12   # output size
sequence_length    = 20   # refers to the amount of timeframes to check
folder_name = ["none", "pinch_in", "pinch_out", "swipe_up", "swipe_down",
               "swipe_left", "swipe_right", "grab2fist", "fist2grab", "peace",
               "2fingers", "pointing"]

# LSTM model
class JarvisLSTM(nn.Module):

    def __init__(self, hidden_dim, input_size, gesture_size, seq_len):
        super(JarvisLSTM, self).__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_len
       
        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, seq_len, batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim, gesture_size)

    def forward(self, seq_batch_input):
        # run the lstm model
        lstm_out, _ = self.lstm(seq_batch_input)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out)
        gesture_out = gesture_out[-1, :, :]

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        #gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        #return gesture_probabilities
        return gesture_out

lstm_model = JarvisLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)
lstm_model.load_state_dict(torch.load("state_dict_model_95_32hidden.pt"))

lstm_model.eval()

# Flask Server stuff
@app.route("/determine-gesture", methods=['POST'])
def determine_gesture():
    json_data = request.get_json()
    data = json.loads(json_data)

    result = {"gesture" : 0}
    return json.dumps(result)

if __name__ == '__main__':
    app.run()
