import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, 'nothing.csv')

number_of_features = 195  # input size
number_of_hidden   = 100  # size of hidden layer
number_of_gestures = 11   # output size
sequence_length    = 10   # refers to the amount of timeframes to check
batch_size         = 5    # how many different files to compute
learning_rate      = 0.0001

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
            if (line_array[i] != "" and line_array[i] != "\n"):
                temp_line.append(float(line_array[i]))
        if len(temp_line) == number_of_features:
            file_contents.append(temp_line)

# convert to a pytorch tensor
training_tensor = torch.FloatTensor(file_contents)
print(training_tensor)
print("Training tensor shape: " + str(training_tensor.shape))

# LSTM model
class JarvisLSTM(nn.Module):

    def __init__(self, hidden_dim, input_size, gesture_size, seq_len):
        super(JarvisLSTM, self).__init__()

        self.seq_length = seq_len
        
        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim, gesture_size)

    def forward(self, seq_batch_input):
        print("forwarding...")
        # run the lstm model
        lstm_out, _ = self.lstm(seq_batch_input)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out)

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        return gesture_probabilities

def train_model(model, training_data, batch_size, loss_function, optimizer, seq_length, num_features):

    # increment by batch_size
    for i in range(seq_length, len(training_data), batch_size):
        # clear the accumulated gradients
        model.zero_grad()

        # convert data into batch_size x num_of_features
        inner_bound = i - seq_length
        sectioned_data = training_data[inner_bound:i, :]
        sectioned_data = torch.unsqueeze(sectioned_data, dim=1)

        for j in range(1, batch_size):
            inner_bound += 1
            temp_matrix = training_data[inner_bound:i + j, :]
            temp_matrix = torch.unsqueeze(temp_matrix, dim=1)
            sectioned_data = torch.cat([sectioned_data, temp_matrix], dim=1)

        # sectioned_data = seq_length x batch x input_size)
        # or time frames, training batch, number of input joint coordinates
        torch.reshape(sectioned_data, (seq_length, batch_size, num_features)) # temporary

        # run forward pass
        resulting_scores = model(sectioned_data)

        print("Resulting_scores: " + str(resulting_scores.shape))
        
        # compute loss and backward propogate
        loss = loss_function(resulting_scores, targets)
        loss.backward()

        optimizer.step()
    
loss_function = nn.NLLLoss()
lstm_model = JarvisLSTM(number_of_hidden, number_of_features, number_of_gestures, sequence_length)
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

train_model(lstm_model, training_tensor, batch_size, loss_function, optimizer, sequence_length, number_of_features)
print("Finished")

