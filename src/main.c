#include <stdlib.h>
#include "test.h"
#include "string.h"
#include "jcr.h"
#include "dubnet.h"

void try_hw0()
{
    srand(0);
    // data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    // data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");
    data train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels");
    data test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels");

    net n = {0};
    n.n = 4;
    n.layers = calloc(n.n, sizeof(layer));
    n.layers[0] = make_connected_layer(3072, 32);
    n.layers[1] = make_activation_layer(RELU);
    // n.layers[2] = make_connected_layer(256, 128);
    // n.layers[3] = make_activation_layer(RELU);
    // n.layers[4] = make_connected_layer(128, 64);
    // n.layers[5] = make_activation_layer(RELU);
    // n.layers[6] = make_connected_layer(64, 32);
    // n.layers[7] = make_activation_layer(RELU);
    n.layers[2] = make_connected_layer(32, 10);
    n.layers[3] = make_activation_layer(SOFTMAX);

    int batch = 128;
    int iters = 1500;
    float rate = .01;
    float momentum = .9;
    float decay = .0005;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
    free_data(train);
    free_data(test);
    free_net(n);
}


// void try_hw1()
// {
//     data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
//     data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

//     net n = {0};
//     n.n = 14;
//     n.layers = calloc(n.n, sizeof(layer));
//     n.layers[0] = make_convolutional_layer(3, 8, 3, 1, 1);
//     n.layers[1] = make_activation_layer(RELU);
//     n.layers[2] = make_maxpool_layer(3, 2);
//     n.layers[3] = make_convolutional_layer(8, 16, 3, 1, 1);
//     n.layers[4] = make_activation_layer(RELU);
//     n.layers[5] = make_maxpool_layer(3, 2);
//     n.layers[6] = make_convolutional_layer(16, 32, 3, 1, 1);
//     n.layers[7] = make_activation_layer(RELU);
//     n.layers[8] = make_maxpool_layer(3, 2);
//     n.layers[9] = make_convolutional_layer(32, 64, 3, 1, 1);
//     n.layers[10] = make_activation_layer(RELU);
//     n.layers[11] = make_maxpool_layer(3, 2);
//     n.layers[12] = make_connected_layer(256, 10);
//     n.layers[13] = make_activation_layer(SOFTMAX);

//     int batch = 128;
//     int iters = 1000;
//     float rate = .01;
//     float momentum = .9;
//     float decay = .005;

//     train_image_classifier(n, train, batch, iters, rate, momentum, decay);
//     printf("Training accuracy: %f\n", accuracy_net(n, train));
//     printf("Testing  accuracy: %f\n", accuracy_net(n, test));
//     free_data(train);
//     free_data(test);
//     free_net(n);
// }


int main(int argc, char **argv)
{
    if(argc < 2){
        printf("usage: %s [test | tryhw0 | tryhw1]\n", argv[0]);  
    } else if (0 == strcmp(argv[1], "tryhw0")){
        try_hw0();
    } else if (0 == strcmp(argv[1], "time")){
        time_matrix_multiply();
    } else if (0 == strcmp(argv[1], "tryhw1")){
        // try_hw1();
    } else if (0 == strcmp(argv[1], "test")){
        if (argc > 2){
            if (0 == strcmp(argv[2], "hw0")){
                test_hw0();
            } else if (0 == strcmp(argv[2], "hw1")){
                test_hw1();
            } else if (0 == strcmp(argv[2], "hw2")){
                test_hw2();
            }
        } else {
            test();
        }
        printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
    }
    return 0;
}
