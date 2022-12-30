from dubnet import *

mnist = 1

inputs = 784 if mnist else 3072

def softmax_model():
    l = [make_connected_layer(inputs, 10),
        make_activation_layer(SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(inputs, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 128),
            make_activation_layer(RELU),
            make_connected_layer(128, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 32),
            make_activation_layer(RELU),
            make_connected_layer(32, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
if mnist:
    train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
    test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
else:
    train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
    test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 4000
rate = .04
momentum = .956
decay = .0005

m = softmax_model()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# !! My M1 Mac was not able to run the python command, 
# so I went to Joseph's OH and told me it was OK to use main.c
# using `./dubnet tryhw0`. So the changes I made to this file
# might not run properly (bc I couldn't test it).
# However, the hyperparams are as listed above, 
# and I used 5-layer neural_net to achieve test accuracy of 0.975400

# Both training and testing accuracies of CIFAR is significantly lower
# than those of MNIST's. After applying changes, MNIST accuracies went
# up, whereas those of CIFAR's dropped.