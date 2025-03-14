import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

# Data prepreprocessing into array of 1x1 matrices
parent_dir = os.path.dirname(__file__)

data = pd.read_csv(os.path.join(parent_dir, "temperature.csv"))
data = data.loc[data['City'] == 'Tokyo']
data = data['AverageTemperatureFahr'].dropna()
data = np.array(data)
data = np.column_stack(data).T
data = (data - 32)*(5/9)

# Split into training, validation, and testing sets
m, n = data.shape
train_data = data[int(m - 0.70 * m):]  # training data
validation_data = data[int(m - 0.85 * m):int(m - 0.70 * m)]  # validation data
testing_data = data[:int(m - 0.85 * m)]  # testing data

#%%
neurons = 50
# Activation functions
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Initialize weights
def init_params():
    Wax = np.random.randn(1, neurons)   # weight of input (1 input, 500 neurons)
    Waa = np.random.randn(neurons, neurons) # weight of hidden layer (500 inputs, 500 neurons)
    ba = np.random.randn(1, neurons)  # bias of hidden layer
    Wab = np.random.randn(neurons, 1) # weight of output layer (500 inputs, 1 output)
    bb = np.random.randn(1, 1) # bias of output layer
    return Wax, Waa, ba, Wab, bb

# Forward pass through the network
def forward_pass(Wax, Waa, ba, Wab, bb, x):
    outputs = np.zeros(len(x))
    hiddens = np.zeros((len(x), neurons))
    prev_hidden = np.zeros((1, neurons))  # Initialize prev_hidden with shape (1, 500)

    for i, temp in enumerate(x):
        x_i = temp.dot(Wax)

        # Ensure prev_hidden is a 1x500 vector for proper matrix multiplication
        x_hidden = x_i + prev_hidden.dot(Waa) + ba
        x_hidden = np.tanh(x_hidden)
       
        prev_hidden = x_hidden  # Update prev_hidden to current hidden state
        hiddens[i, :] = x_hidden.flatten()  # Ensure hidden is a 1D array for each time step
       
        x_output = x_hidden.dot(Wab) + bb
        outputs[i] = x_output.flatten()  # Flatten output to match the correct shape

    return outputs, hiddens

# Mean Squared Error
def MSE(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def MSE_grad(actual, predicted):
    return (predicted - actual)

# Backward pass to update the weights
def backward_pass(actual, predicted, hiddens, Wax, Waa, ba, Wab, bb):
    next_hidden = np.zeros((1, neurons))  # Initialize next_hidden with correct shape
    output_weight_grad, output_bias_grad, hidden_weight_grad, hidden_bias_grad, input_weight_grad = [0] * 5
    loss_grad = MSE_grad(actual, predicted)

    for i in range(len(hiddens) - 1, -1, -1):
        l_grad = loss_grad[i]  # Convert loss_grad to match the shape

        output_weight_grad += hiddens[i].T.reshape(-1, 1) * l_grad  # Reshape hiddens[i] to (500, 1)
        output_bias_grad += np.mean(loss_grad)

        output_grad = l_grad * Wab.T
        if next_hidden is None:
            hidden_grad = output_grad
        else:
            hidden_grad = output_grad + next_hidden.dot(Waa.T)

        tanh_deriv = 1 - hiddens[i, :]**2
        hidden_grad = hidden_grad * tanh_deriv

        next_hidden = hidden_grad

        if i > 0:
            hidden_weight_grad += hiddens[i - 1, :].T.reshape(-1, 1).dot(hidden_grad)
            hidden_bias_grad += np.mean(hidden_grad)

        input_weight_grad += actual[i] * hidden_grad

    learning_rate = 1e-5

    # Update weights using the gradients
    Wax -= input_weight_grad * learning_rate
    Waa -= hidden_weight_grad * learning_rate
    ba -= hidden_bias_grad * learning_rate
    Wab -= output_weight_grad * learning_rate
    bb -= output_bias_grad * learning_rate

    return Wax, Waa, ba, Wab, bb

# Training Loop
def train_rnn(train_data, num_epochs, batch_size):
    # Initialize parameters
    Wax, Waa, ba, Wab, bb = init_params()
    losses = []

    # Iterate through epochs
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Loop through batches of training data
        for i in range(0, len(train_data) - batch_size, batch_size):
            batch_data = train_data[i:i+batch_size]
            # Forward pass
            outputs, hiddens = forward_pass(Wax, Waa, ba, Wab, bb, batch_data)
            # Backward pass
            Wax, Waa, ba, Wab, bb = backward_pass(batch_data[:, 0], outputs, hiddens, Wax, Waa, ba, Wab, bb)
            # Calculate loss for this batch
            batch_loss = MSE(batch_data[:, 0], outputs)
            epoch_loss += batch_loss

        # Store average loss for the epoch
        losses.append(epoch_loss / len(train_data))
       
        # Print the loss for every 10th epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss / len(train_data)}')

    # Plot the loss curve
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    return Wax, Waa, ba, Wab, bb

# Train the RNN
np.random.shuffle(train_data)
Wax, Waa, ba, Wab, bb = train_rnn(train_data, num_epochs = 200, batch_size = 32)


#%%
# Testing the model
# data = data
mixed_data = data[20:50]
np.random.shuffle(mixed_data)
outputs, hiddens = forward_pass(Wax, Waa, ba, Wab, bb, mixed_data)
print("Testing Loss:", np.round(MSE(mixed_data[:, 0], outputs), 2))
print(f"Actual Errors: {np.round(MSE_grad(mixed_data[:, 0], outputs), 2)}")

x = [i for i in range(len(outputs))]
plt.figure(figsize = (16, 5))
plt.plot(x,
         outputs,
         'r--',
         label = 'predicted')
plt.plot(x,
         mixed_data,
         label = 'actual',
         marker = 'o',
         lw = 0.0,
         mfc = None,
         mec = 'b')
plt.xlabel('time (idx)')
plt.ylabel(u'temp (\u2109)')
plt.title('Weather in Tokyo')
plt.legend()
plt.show()

#%%

def predict_future(Wax, Waa, ba, Wab, bb, input_sequence, num_predictions):
    predictions = []
    current_input = input_sequence
   
    for _ in range(num_predictions):
        output, _ = forward_pass(Wax, Waa, ba, Wab, bb, current_input)
       
        predictions.append(output[int(len(output)/2)])  # Append the last predicted value
       
        # Use the last predicted value to create the next input sequence
        current_input = np.roll(current_input, -1, axis = 0)  # Shift the sequence
        current_input[-1] = output[-1]  # Add the new prediction to the sequence

    return predictions

input_sequence = data[-100:]
# np.random.shuffle(input_sequence)
num_predictions = 30

predicted_temps_future = predict_future(Wax, Waa, ba, Wab, bb, input_sequence, num_predictions)
predicted_temps_past, hiddens = forward_pass(Wax, Waa, ba, Wab, bb, input_sequence)

predicted_altogether = np.append(predicted_temps_past, predicted_temps_future)

x_idx = [i for i in range(len(predicted_altogether))]

plt.figure(figsize = (8, 3))
plt.grid()
plt.plot(x_idx,
         predicted_altogether,
         marker = 'o',
         mfc = 'none',
         lw = 0.0,
         label = 'prediction')
plt.plot(x_idx[0:len(input_sequence)],
         # data[0:(len(input_sequence) + num_predictions)],
         input_sequence,
         'r--',
         label = 'actual')
plt.yticks(np.arange(10, 40, 5))
plt.xticks(np.arange(0, np.max(x_idx) + 10, 10))
plt.xlabel('Time (idx)')
plt.ylabel('Temperature (°F)')
plt.title('Temperature Prediction of Tokyo')
plt.legend()
plt.show()