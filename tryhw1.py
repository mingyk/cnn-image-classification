from dubnet import *

def conv_net():
    l = [   make_convolutional_layer(3, 8, 3, 1), # 32 * 32 * 3 * 8 * 9 = 221184
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(8, 16, 3, 1), # 15 * 15 * 8 * 16 * 9 = 259200
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(16, 32, 3, 1), # 7 * 7 * 16 * 32 * 9 = 225792
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_convolutional_layer(32, 64, 3, 1), # 3 * 3 * 32 * 64 * 9 = 165888
            make_activation_layer(RELU),
            make_maxpool_layer(3, 2),
            make_connected_layer(256, 10), # 1 * 1 * 256 * 10 = 2560
            make_activation_layer(SOFTMAX)]
    return make_net(l) # Computation: 874624

def neural_net():
    l = [   make_connected_layer(3072, 256), # 786432
            make_activation_layer(RELU),
            make_connected_layer(256, 128), # 32768
            make_activation_layer(RELU),
            make_connected_layer(128, 64), # 8192
            make_activation_layer(RELU),
            make_connected_layer(64, 32), # 2048
            make_activation_layer(RELU),
            make_connected_layer(32, 10), # 320
            make_activation_layer(SOFTMAX)]
    return make_net(l) # Computation: 829760

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# The training accuracy of the fully connected network is 0.4 and its test accuraacy is 0.399.
# In comparison, both of the training accuracy and the testing accuracy of the convnet are 0.533.
# Both network uses the similar number of operations (829760 v. 874624), but the convnet result is higher
# because the network goes through each pixel through convolutional and maxpooling layers, which makes
# the performance significantly better.

