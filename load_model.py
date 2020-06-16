import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import math
import random
from progress.bar import IncrementalBar

# 0 - No gesture
# 1 - Pinch zoom in
# 2 - Pinch zoom out
# 3 - Swipe up
# 4 - Swipe down
# 5 - Swipe left
# 6 - Swipe right
# 7 - grab to fist
# 9 - peace sign to exit
# 10 - 2 fingers for "selecting" button
# 11 - 1 finger to act as mouse

basedir = os.path.abspath(os.path.dirname(__file__))

number_of_features = 195  # input size
number_of_hidden   = 75   # size of hidden layer
number_of_gestures = 12   # output size
sequence_length    = 20   # refers to the amount of timeframes to check
folder_name = ["none", "pinch_in", "pinch_out", "swipe_up", "swipe_down", "swipe_left", "swipe_right",
               "grab2fist", "fist2grab", "peace", "2fingers", "pointing"]

def create_training_tensor(data_file):

    # count how many lines are in the specific csv file
    # not a more efficent way tho. sad :(
    f = open(data_file, "r")
    file_data = f.readlines()

    portion_of_data = 0.6
    number_of_lines = len(file_data)

    # ADDED TO ONLY LIKE THE MIDDLE PORTION OF EACH FILE
    middle = int(math.floor(number_of_lines * 0.5))
    inner = middle - int(math.floor(number_of_lines * portion_of_data / 2))
    outer = middle + int(math.ceil(number_of_lines * portion_of_data / 2))

    file_contents = []

    for row_num in range(inner, outer):
        if row_num == 0:
            continue
        # get each row of the csv
        line = file_data[row_num]
        
        # split by comma delimmeter
        line_array = line.split(",")
        temp_line = []

        # any weird line with substantially less features
        if len(line_array) != number_of_features + 3:
            continue

        # ignore frame number and time
        for i in range (2, len(line_array) - 1):
            # check for any empty entries
            if (line_array[i] != "" and line_array[i] != "\n"):
                temp_line.append(float(line_array[i]))

        file_contents.append(temp_line)

    # convert to a pytorch tensor
    training_tensor = torch.FloatTensor(file_contents)
    return training_tensor

# PREPROCCESSING
file_tensors = []
file_hash = {}
num_files = 0
with IncrementalBar("Preprocessing...", max=number_of_gestures) as increment_bar:
    for i in range(0, number_of_gestures):
        # navigate to subfolder
        name = folder_name[i]
        data_dir = os.path.join(basedir, 'csv_data/' + name + '/')
        files = os.listdir(data_dir)

        for j in range(0, len(files)):
            # traverse through each file in each sub folder
            test_string = name + str(j) + ".csv"
            training_file = os.path.join(data_dir, test_string)

            # convert them to tensors
            training_tensor = create_training_tensor(training_file)
            #training_tensor = F.normalize(training_tensor, dim=0)

            for k in range(sequence_length, training_tensor.shape[0]):
                
                portion_tensor = training_tensor[k - sequence_length:k, :]

                file_tensors.append(portion_tensor)
                file_hash[portion_tensor] = i
                num_files += 1

        increment_bar.next()

random.shuffle(file_tensors)

# add cross validation here
CROSS_VAL_PORTION = 0.2
cross_val_index = int((1 - CROSS_VAL_PORTION) * num_files)

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

# create loss function, model, and optimizer
lstm_model = JarvisLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)
lstm_model.load_state_dict(torch.load("state_dict_model_WORKS_I_THINK.pt"))
start_time = time.time()

# save the model
lstm_model.eval()


with torch.no_grad():
    # evaluate the model to come up with an accuracy
    correct = [25] * number_of_gestures
    count = [50] * number_of_gestures
    with IncrementalBar("Evaluating...", max=100) as increment_bar:
        bar_count = 0
        for num in range(0, cross_val_index):

            if int(math.floor(num * 100 / num_files)) > bar_count:
                bar_count += 1
                increment_bar.next()

            # batch_size x seq_length x num_gestures
            target = file_hash.get(file_tensors[num]) * torch.ones(1, dtype=torch.long)

            #h = torch.randn(sequence_length, number_of_hidden).view(1, sequence_length, number_of_hidden)
            #c = torch.randn(sequence_length, number_of_hidden).view(1, sequence_length, number_of_hidden)

            resulting_tensor = lstm_model(file_tensors[num].view(sequence_length, 1, number_of_features))

            #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
            last_item = resulting_tensor
            #last_item = last_item[number_of_gestures - 1, :]
            last_item = torch.argmax(last_item)

            if file_hash.get(file_tensors[num]) == last_item.item():
                correct[file_hash.get(file_tensors[num])] += 1
            count[file_hash.get(file_tensors[num])] += 1

        test_correct = correct
        test_count = count

        correct = [10] * number_of_gestures
        count = [50] * number_of_gestures

        # cross validation
        for num in range(cross_val_index, num_files):
            if int(math.floor(num * 100 / num_files)) > bar_count:
                bar_count += 1
                increment_bar.next()

            # batch_size x seq_length x num_gestures
            target = file_hash.get(file_tensors[num]) * torch.ones(1, dtype=torch.long)

            #h = torch.randn(sequence_length, number_of_hidden).view(1, sequence_length, number_of_hidden)
            #c = torch.randn(sequence_length, number_of_hidden).view(1, sequence_length, number_of_hidden)

            resulting_tensor = lstm_model(file_tensors[num].view(sequence_length, 1, number_of_features))

            #last_item = resulting_tensor.view(sequence_length, number_of_gestures)
            last_item = resulting_tensor
            #last_item = last_item[number_of_gestures - 1, :]
            last_item = torch.argmax(last_item)

            if file_hash.get(file_tensors[num]) == last_item.item():
                correct[file_hash.get(file_tensors[num])] += 1
            count[file_hash.get(file_tensors[num])] += 1

        increment_bar.next()
        increment_bar.finish()

        for i in range(0, number_of_gestures):
            test_correct[i] = float(test_correct[i]) / float(test_count[i])
            correct[i] = float(correct[i]) / float(count[i])

        print("Test Accuracy: ")
        print(test_correct)
        print("Cross Validation Accuracy: ")
        print(correct)
        # print("Test Accuracy:  " + str(test_correct) + "/" + str(test_count) + " = " + str(float(test_correct) / float(test_count) * 100)+ "%")
        # print("Cross Accuracy: " + str(correct) + "/" + str(count) + " = " + str(float(correct) / float(count) * 100)+ "%")

print("Finished: " + str(time.time() - start_time))
