import numpy as np
import scipy.io
from nn import *
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 150
batch_size = 32
learning_rate = 0.002
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)
params = {}
valid_eg = valid_x.shape[0]

initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initial_W = params['Wlayer1']
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

loss_train, loss_test, acc_train, acc_test = [], [], [], []

rows, cols = params['Wlayer1'].shape
#fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))

# Iterate for all subplots
#for i, ax in enumerate(axes.flat):
#    ax.imshow(params['Wlayer1'][:, i].reshape((32, 32)))

#plt.show()

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_acc += acc
        total_loss += loss

        # backward
        delta = probs - yb
        grad = backwards(delta, params, 'output', linear_deriv)
        backwards(grad, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] = params['Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['blayer1'] = params['blayer1'] - learning_rate * params['grad_blayer1']
        params['Woutput'] = params['Woutput'] - learning_rate * params['grad_Woutput']
        params['boutput'] = params['boutput'] - learning_rate * params['grad_boutput']

    avg_acc = total_acc / batch_num
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# run on validation set and report accuracy! should be above 75%
    # Forward pass through the neural network layers
    v_h1 = forward(valid_x, params, 'layer1')
    v_probs = forward(v_h1, params, 'output', softmax)

    # Compute loss and accuracy for test set
    v_loss, vacc = compute_loss_and_acc(valid_y, v_probs)

    # Store training and validation metrics
    loss_train.append(total_loss / train_x.shape[0])
    loss_test.append(v_loss / valid_eg)
    acc_train.append(avg_acc)
    acc_test.append(vacc)

# Print final accuracy
last_accuracy = acc_test[-1]
last_accuracy *= 100
print(f'Validation accuracy: {last_accuracy:.2f}')

# Plotting training and validation metrics
plt.figure(0)
plt.plot(np.arange(max_iters), acc_train, 'r')
plt.plot(np.arange(max_iters), acc_test, 'b')
plt.legend(['Training Accuracy', 'Testing Accuracy'])

plt.figure(1)
plt.plot(np.arange(max_iters), loss_train, 'r')
plt.plot(np.arange(max_iters), loss_test, 'b')
plt.legend(['Training Loss', 'Testing Loss'])
plt.show()


# Save trained neural network parameters
saved_params = {k: v for k, v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
rows, cols = params['Wlayer1'].shape
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))

# Iterate for all subplots
for i, ax in enumerate(axes.flat):
    ax.imshow(params['Wlayer1'][:, i].reshape((32, 32)))

plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

h1 = forward(train_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)

predicted_classes = np.argmax(probs, axis=1)
true_classes = np.argmax(train_y, axis=1)

np.add.at(confusion_matrix, (predicted_classes, true_classes), 1)

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()