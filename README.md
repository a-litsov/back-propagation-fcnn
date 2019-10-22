### Deep Learning. Individual Laboratory Work.

## How to run
1. Install [JRE 8+](https://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html), but I believe
you already have it installed.
2. **Just take `fcnn-a-litsov.jar` file from release** or build it by yourself using the following steps:
    1. Install [JDK 8+](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
    2. Install [Apache Maven 3+](http://maven.apache.org/download.cgi).
    3. In repository root run the following command `mvn clean install`.
    4. Take `fcnn-a-litsov.jar` file from `/target` directory.
3. Run the program with the following command:
```bash
java -jar fcnn-a-litsov.jar [learning_rate] [epoch] [hidden layer size] [batch size]
```

## Theory
All theory is located in `/theory` folder as images (ru).

## Results
| Learning rate | Epochs | Hidden layer | Batch Size | Test Accuracy, (%)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.1  | 10  | 16 | 10 | 94.04 |
| 0.05  | 10  | 16 | 10 | 94.17 |
| 0.01  | 10  | 16 | 10 | 93.69 |
| 0.05  | 10  | 80 | 10 | 97.21 |
| 0.05  | 20  | 80 | 10 | **97.43** |
| 0.05  | 20  | 80 | 100 | 96.27 |
| 0.05  | 20  | 80 | 256 | 94.8 |

## About
Two-layer fully-connected neural network with tanh activation function used in the hidden layer and softmax activation
function used in the output layer.

## Dataset info
THE MNIST DATABASE of handwritten digits is used.

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test
set of 10,000 examples.
It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size 28x28 image.


