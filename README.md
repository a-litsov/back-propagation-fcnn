### Deep Learning. Individual Laboratory Work.

## How to run
1. Install [JRE 8+](https://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html), but I believe
you already have it installed.
2. **Just take jar file from release** or build it by yourself using the following steps:
    0. Install [JDK 8+](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
    1. Install [Apache Maven 3+](http://maven.apache.org/download.cgi).
    2. In repository root run the following command `mvn clean install`.
    3. Desired `fcnn-a-litsov.jar` file appeared in `/target` directory.
3. Run the program with the following command:
```bash
java -jar fcnn-a-litsov.jar [learning_rate] [epoch] [hidden layer size] [batch size]
```

## Theory
All theory is located in `/theory` folder as images (ru).

## About
Two-layer fully-connected neural network with tanh activation function used in the hidden layer and softmax activation
function used in output layer.

## Dataset info
THE MNIST DATABASE of handwritten digits is used.

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test
set of 10,000 examples.
It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size 28x28 image.


