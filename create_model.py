import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# LSTM model
class JarvisLSTM(nn.Module):

    def __init__(self, hidden_dim, input_size, gesture_size):
        super(JarvisLSTM, self).__init__()

        # create the LSTM network
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=False)

        # linear space maps between hidden layer and output layer
        self.hidden2gesture = nn.Linear(hidden_dim, gesture_size)

    def forward(self, seq_batch_input, seq_length):
        # run the lstm model
        lstm_out, _ = self.lstm(seq_batch_input)

        # convert to the proper output dimensions
        gesture_out = self.hidden2gesture(lstm_out.view(len(seq_length), -1))

        # apply softmax function to normalize the values
        # double check dimension. prob wrong
        gesture_probabilities = F.log_softmax(gesture_out, dim=1)
        return gesture_probabilities


number_of_gestures = 11   # output size
sequence_length = 10      # however long 1 second is

learning_rate = 0.0001


loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, training_data, loss_function, optimizer):

    # clear the accumulated gradients
    model.zero_grad()

    #TODO: change training_data in such a way
    # some_parameter = training_data + some manipulation

    # run forward pass
    # some_parameter = (seq_length, batch, input_size)
    # or time frames, training batch, number of input joint coordinates
    # resulting_scores = model(some_parameter)

    # compute loss and backward propogate
    # loss = loss_function(resulting_scores, targets)
    # loss.backward()

    optimizer.step()
    
