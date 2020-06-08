import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import math
from progress.bar import IncrementalBar

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

basedir = os.path.abspath(os.path.dirname(__file__))

number_of_features = 192  # input size
number_of_hidden   = 100  # size of hidden layer
number_of_gestures = 12   # output size
sequence_length    = 20   # refers to the amount of timeframes to check
batch_size         = 5    # how many different files to compute
learning_rate      = 0.001
num_epoch          = 50
folder_name = ["none", "pinch_in", "pinch_out", "swipe_up", "swipe_down", "swipe_left", "swipe_right",
               "grab2fist", "fist2grab", "peace", "2fingers", "pointing"]

# PREPROCCESSING
#file_tensors = []
#for i in range(0, len(folder_name)):

def create_training_tensor(data_file):
    # construct a matrix tensor that has the proper dimensions
    # should be num_of_features x max_batch_size
    file_contents = []
    with open(data_file, "r") as file:
        file.readline()
        for line in file.readlines():
            line_array = line.split(",")
            temp_line = []
            # ignore frame number and time
            for i in range (2, len(line_array) - 1):
                # check for any empty entries
                if (line_array[i] != "" and line_array[i] != "\n" and line_array[i] != "0"):
                    temp_line.append(float(line_array[i]))
            if len(temp_line) == number_of_features:
                file_contents.append(temp_line)

    # convert to a pytorch tensor
    training_tensor = torch.FloatTensor(file_contents)
    return training_tensor

# LSTM model
class JarvisLSTM(nn.Module):

    def __init__(self, hidden_dim, input_size, gesture_size, seq_len):
        super(JarvisLSTM, self).__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_len
       
        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim, gesture_size)

    def forward(self, seq_batch_input, hc):
        # run the lstm model
        lstm_out, _ = self.lstm(seq_batch_input, hc)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out)

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        #gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        #return gesture_probabilities
        return gesture_out

def train_model(model, training_data, batch_size, loss_function, optimizer, seq_length, num_features, targets, hidden_dim):

    portion_of_data = 0.6
    running_loss = 0.0
    count = 0

    # ADDED TO ONLY DO LIKE THE MIDDLE PORTION OF EACH FILE
    iterations = (((len(training_data) - batch_size)) - seq_length) / batch_size

    middle = int(math.floor(iterations * 0.5))
    inner = middle - int(math.floor(iterations * portion_of_data / 2))
    outer = middle + int(math.ceil(iterations * portion_of_data / 2))

    # iterate through the portions
    for j in range(inner, outer):

        i = (j * batch_size) + seq_length

        # convert data into batch_size x num_of_features
        inner_bound = i - seq_length
        sectioned_data = training_data[inner_bound:i, :]
        sectioned_data = torch.unsqueeze(sectioned_data, dim=1)

        # concatenate the matrix together to be the proper size
        for k in range(1, batch_size):
            inner_bound += 1
            temp_matrix = training_data[inner_bound:i + k, :]
            temp_matrix = torch.unsqueeze(temp_matrix, dim=1)
            sectioned_data = torch.cat([sectioned_data, temp_matrix], dim=1)

        # sectioned_data = seq_length x batch x input_size)
        # or time frames, training batch, number of input joint coordinates
        sectioned_data = torch.reshape(sectioned_data, (batch_size, seq_length, num_features)) # temporary

        h = torch.randn(seq_length, hidden_dim).view(1, seq_length, hidden_dim)
        c = torch.randn(seq_length, hidden_dim).view(1, seq_length, hidden_dim)

        # run forward pass
        resulting_scores = model(sectioned_data, (h, c))

        # clear the accumulated gradients
        model.zero_grad()
        optimizer.zero_grad()

        # compute loss and backward propogate
        loss = loss_function(resulting_scores, targets)
        print(str(loss) + " " + str(loss.item()))
        loss.backward()

        optimizer.step()
        count += 1
        running_loss += loss.item()
        #print(str(list(model.parameters())[0].grad))

    if (count == 0):
        return 0
    return running_loss
   
def epoch(folder_name, number_of_gestures, batch_size, lstm_model, loss_function, optimizer, sequence_length, number_of_features, epoch_num, hidden_dim):
    avg_total_loss = []
    
    # traverse through all of the 12 gesture training data
    with IncrementalBar("Training " + str(epoch_num) + "...", max=number_of_gestures) as increment_bar:
        for i in range(0, number_of_gestures):
            name = folder_name[i]
            print(name)

            basedir = os.path.abspath(os.path.dirname(__file__))
            data_dir = os.path.join(basedir, 'csv_data/' + name + '/')
            files = os.listdir(data_dir)
            file_hash = {}

            # turn the file data into a hash for O(1)
            for j in range(0, len(files)):
                file_hash[files[j]] = 0

            count = 0

            # batch_size x seq_length x num_gestures
            target = i * torch.ones(batch_size, number_of_gestures, dtype=torch.long)

            # determine what count number this is
            test_string = name + str(count) + ".csv"

            return_loss = 0.0

            while True:
                if test_string in file_hash:
                    # csv file exists. therefore train with it
                    training_file = os.path.join(data_dir, test_string)
                    training_tensor = create_training_tensor(training_file)
                    return_loss = train_model(lstm_model, training_tensor, batch_size, loss_function,
                                           optimizer, sequence_length, number_of_features, target, hidden_dim)
                    count += 1
                    test_string = name + str(count) + ".csv"
                else:
                    break
            increment_bar.next()
            avg_total_loss.append(return_loss)
            
    print("Loss: " + str((avg_total_loss)))
    increment_bar.finish()

loss_function = nn.CrossEntropyLoss()
lstm_model = JarvisLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
start_time = time.time()

with IncrementalBar("Epoch...", max=num_epoch, suffix='%(percent).1f%% - %(eta)ds') as bar:
    lstm_model.train()
    for i in range (0, num_epoch):
        epoch(folder_name, number_of_gestures, batch_size, lstm_model, loss_function, optimizer, sequence_length, number_of_features, i, number_of_hidden)
        #for param in lstm_model.parameters():
        #    print(param.data)
        bar.next()

STORAGE_PATH = "state_dict_model.pt"

torch.save(lstm_model.state_dict(), STORAGE_PATH)

print("Finished: " + str(time.time() - start_time))
