package ru.unn.itmm.fcnn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.unn.itmm.fcnn.util.Utils;

class Network {
    private static final Logger logger = LoggerFactory.getLogger(Network.class);

    private float avg;

    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    private float[][] input;
    private float[][] hidden;
    private float[][] output;

    private float[][] hiddenWeights; // weights between input and hidden layer
    private float[][] outputWeights; // weights between hidden layer and output layer

    private float[][] hiddenDerivatives;
    private float[][] outputDerivatives;

    private float outputMax;

    Network(int batchSize, int inputSize, int hiddenSize, int outputSize) {
        logger.info("Started network configuration with {} batch size, {} input neurons, {} hidden neurons and "
                + "{} output neurons", batchSize, inputSize, hiddenSize, outputSize);
        this.avg = (float) 1 / batchSize;

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        input = new float[batchSize][inputSize];
        hidden = new float[batchSize][hiddenSize];
        output = new float[batchSize][outputSize];

        hiddenWeights = new float[hiddenSize][inputSize + 1]; // extra element is for threshold
        Utils.initRandomWeights(hiddenWeights);
        outputWeights = new float[outputSize][hiddenSize + 1];
        Utils.initRandomWeights(outputWeights);

        hiddenDerivatives = new float[batchSize][hiddenSize];
        outputDerivatives = new float[batchSize][outputSize];

        logger.info("Network configuration done, gradient matrices will be summed with {} coefficient", avg);
    }

    private void singleForwardPass(float[] input, float[] hidden, float[] hiddenDerivatives, float[] output) {
        logger.debug("singleForwardPass");
        for (int i = 0; i < hiddenSize; i++) {
            hidden[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                hidden[i] += hiddenWeights[i][j] * input[j];
            }
            hidden[i] += hiddenWeights[i][inputSize]; // adding threshold

            // end of hidden calculation
            hidden[i] = (float) Math.tanh(hidden[i]);
            hiddenDerivatives[i] = Utils.tanhDerivative(hidden[i]);
        }

        outputMax = 0;
        for (int i = 0; i < outputSize; i++) {
            output[i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += outputWeights[i][j] * hidden[j];
            }
            output[i] += outputWeights[i][hiddenSize]; // adding threshold

            if (outputMax < output[i]) {
                outputMax = output[i];
            }
        }
        outputMax /= 2;

        for (int i = 0; i < outputSize; i++) {
            output[i] = Utils.softMax(output, outputMax, i);
        }
        logger.debug("singleForwardPass done");
    }

    private void batchForwardPass(float[][] input, int realSize) {
        logger.debug("batchForwardPass");
        for (int i = 0; i < realSize; i++) {
            singleForwardPass(input[i], hidden[i], hiddenDerivatives[i], output[i]);
        }
        logger.debug("batchForwardPass ended");
    }

    private void backwardPass(
            int label, float[] input, float[] output, float[] hidden, float[] outputDerivatives,
            float[] hiddenDerivatives, float learningRate
    ) {
        logger.debug("started single backward pass");
        for (int i = 0; i < outputSize; i++) {
            outputDerivatives[i] = (i == label) ? output[i] - 1.0f : output[i];
            for (int j = 0; j < hiddenSize; j++) {
                outputWeights[i][j] -= outputDerivatives[i] * hidden[j] * learningRate * avg;
            }
            outputWeights[i][hiddenSize] -= outputDerivatives[i] * learningRate * avg;
        }


        for (int i = 0; i < hiddenSize; i++) {
            float sum = 0;
            for (int j = 0; j < outputSize; j++) {
                 sum += outputDerivatives[j] * outputWeights[j][i];
            }
            hiddenDerivatives[i] *= sum;
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                hiddenWeights[i][j] -= hiddenDerivatives[i] * input[j] * learningRate * avg;
            }
            hiddenWeights[i][inputSize] -= hiddenDerivatives[i] * learningRate * avg;
        }
        logger.debug("single backward pass ended");
    }

    void teach(float[][] input, int realSize, int[] labels, float learningRate) {
        this.input = input;

        batchForwardPass(input, realSize);
        batchBackwardPass(labels, realSize, learningRate);
    }

    private void batchBackwardPass(int[] labels, int realSize, float learningRate) {
        logger.debug("started batch backward pass");
        for (int i = 0; i < realSize; i++) {
            backwardPass(labels[i], input[i], output[i], hidden[i], outputDerivatives[i], hiddenDerivatives[i], learningRate);
        }
        logger.debug("batch backward pass ended");
    }

    int predict() {
        logger.debug("started predicting");
        float max = output[0][0];
        int value = 0;
        for (int i = 0; i < outputSize; i++)
            if (output[0][i] > max) {
                max = output[0][i];
                value = i;			//predicted class
            }
        logger.debug("predicting finished. predicted class {}, it's probability {}", value, max);
        return value;
    }

    int test(float[] input) {
        logger.debug("started testing");
        singleForwardPass(input, hidden[0], hiddenDerivatives[0], output[0]);
        logger.debug("testing ended");
        return predict();
    }
}
