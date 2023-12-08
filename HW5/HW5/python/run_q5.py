import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

train_loss_arr = []
test_loss_arr = []
train_accuracy_arr = []
test_accuracy_arr = []

max_iters = 150
batch_size = 32
learning_rate = 0.002
hidden_size = 64
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
# 1024 to 32
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
# 32 to 32
initialize_weights(hidden_size, hidden_size, params, 'hidden1')
# 32 to 32
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
#32 to 1024
initialize_weights(hidden_size, train_x.shape[1], params, 'output')

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # Forwad passes for layer using relu 
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden1', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)

        # Find loss from xb initial to output
        loss = np.sum((xb - out)**2)
        total_loss += loss

        # All backward passes to calculate gradients
        d1 = 2 * (out - xb)
        d2 = backwards(d1, params, 'output', sigmoid_deriv)
        d3 = backwards(d2, params, 'hidden2', relu_deriv)
        d4 = backwards(d3, params, 'hidden1', relu_deriv)
        backwards(d4, params, 'layer1', relu_deriv)

        # Use momentum to update gradient descent
        params['m_Wlayer1'] = 0.9 * params['m_Wlayer1']-learning_rate*params['grad_Wlayer1']
        params['Wlayer1'] += params['m_Wlayer1']
        
        params['m_Wlayer2'] = 0.9 * params['m_Wlayer2']-learning_rate*params['grad_Wlayer2']
        params['Wlayer2'] += params['m_Wlayer2']
        
        params['m_Wlayer3'] = 0.9 * params['m_Wlayer3']-learning_rate*params['grad_Wlayer3']
        params['Wlayer3'] += params['m_Wlayer3']
        
        params['m_Woutput'] = 0.9 * params['m_Woutput']-learning_rate*params['grad_Woutput']
        params['Woutput'] += params['m_Woutput']
        
        params['m_blayer1'] = 0.9 * params['m_blayer1']-learning_rate*params['grad_blayer1']
        params['blayer1'] += params['m_blayer1']
        
        params['m_blayer2'] = 0.9 *  params['m_blayer2']-learning_rate*params['grad_blayer2']
        params['blayer2'] += params['m_blayer2']
         
        params['m_blayer3'] = 0.9 * params['m_blayer3']-learning_rate*params['grad_blayer3']
        params['blayer3'] += params['m_blayer3']
        
        params['m_boutput'] = 0.9 * params['m_boutput']-learning_rate*params['grad_boutput']
        params['boutput'] += params['m_boutput']

    # Testing pass through the network layers
    test_h1 = forward(valid_x, params, 'layer1', relu)
    test_h2 = forward(test_h1, params, 'hidden1', relu)
    test_h3 = forward(test_h2, params, 'hidden2', relu)
    test_out = forward(test_h3, params, 'output', sigmoid)
    test_loss = np.sum((valid_x - test_out) ** 2)

    train_loss_arr.append(total_loss / train_x.shape[0])
    test_loss_arr.append(test_loss / valid_x.shape[0])

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9


plt.figure(0)
plt.plot(np.arange(max_iters), train_loss_arr, 'r', label='Train Loss')  
plt.plot(np.arange(max_iters), test_loss_arr, 'b', label='Test or Valid Loss') 
plt.legend() 
plt.xlabel('Iterations')  
plt.ylabel('Loss') 
plt.title('Training and Test/Validation Loss')  
plt.show() 

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
h1 = forward(xb, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden1', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)
print(out.shape)
print(valid_x.shape)
for i in range(5):
    plt.subplot(1, 2, 1)
    plt.imshow(xb[i].reshape(32, 32).T)
    plt.subplot(1, 2, 2)
    plt.imshow(out[i].reshape(32, 32).T)
    plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
valid_h1 = forward(valid_x, params, 'layer1', relu)
valid_h2 = forward(valid_h1, params, 'hidden1', relu)
valid_h3 = forward(valid_h2, params, 'hidden2', relu)
valid_out = forward(valid_h3, params, 'output', sigmoid)

psnr_values = []
for i in range(len(valid_x)):
    psnr = peak_signal_noise_ratio(valid_x[i], valid_out[i])
    psnr_values.append(psnr)

average_psnr = np.mean(psnr_values)

print(f"Average PSNR across all validation images: {average_psnr}")


