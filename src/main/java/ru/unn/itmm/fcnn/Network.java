package ru.unn.itmm.fcnn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.unn.itmm.fcnn.util.Utils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

class Network {
    private static final Logger logger = LoggerFactory.getLogger(Network.class);

    private static final int THREADS = 8;

    private ExecutorService executorService = Executors.newFixedThreadPool(THREADS);

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

    Network(int batchSize, int inputSize, int hiddenSize, int outputSize) {
        logger.info("Started network configuration with {} hidden neurons, {} batch size, {} input neurons, and "
                + "{} output neurons", hiddenSize, batchSize, inputSize, outputSize);
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

        logger.info("Network initial configuration is done");
    }

    /**
     * @param current - position of current sample in batch array
     */
    private void singleForwardPass(int current) {
        logger.debug("singleForwardPass");
        for (int i = 0; i < hiddenSize; i++) {
            hidden[current][i] = 0;
            for (int j = 0; j < inputSize; j++) {
                hidden[current][i] += hiddenWeights[i][j] * input[current][j];
            }
            hidden[current][i] += hiddenWeights[i][inputSize]; // adding threshold

            // end of hidden calculation
            hidden[current][i] = (float) Math.tanh(hidden[current][i]);
            hiddenDerivatives[current][i] = Utils.tanhDerivative(hidden[current][i]);
        }

        float outputMax = 0;
        for (int i = 0; i < outputSize; i++) {
            output[current][i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                output[current][i] += outputWeights[i][j] * hidden[current][j];
            }
            output[current][i] += outputWeights[i][hiddenSize]; // adding threshold

            if (outputMax < output[current][i]) {
                outputMax = output[current][i];
            }
        }
        outputMax /= 2;

        for (int i = 0; i < outputSize; i++) {
            output[current][i] = Utils.softMax(output[current], outputMax, i);
        }
        logger.debug("singleForwardPass done");
    }

    private void batchForwardPass(int realSize) {
        logger.debug("batchForwardPass");
        IntStream.range(0, realSize).parallel().forEach(
                this::singleForwardPass
        );
        logger.debug("batchForwardPass ended");
    }

    /**
     * @param current - position of current sample in batch array
     */
    private void backwardPass(int current, int label, float learningRate) {
        logger.debug("started single backward pass");
        System.arraycopy(output[current], 0, outputDerivatives[current], 0, output[current].length);
        outputDerivatives[current][label] -= 1;
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                outputWeights[i][j] -= outputDerivatives[current][i] * hidden[current][j] * learningRate * avg;
            }
            outputWeights[i][hiddenSize] -= outputDerivatives[current][i] * learningRate * avg;
        }

        IntStream.range(0, hiddenSize).parallel().forEach(i -> {
            float sum = 0;
            for (int j = 0; j < outputSize; j++) {
                 sum += outputDerivatives[current][j] * outputWeights[j][i];
            }
            hiddenDerivatives[current][i] *= sum;

            for (int j = 0; j < inputSize; j++) {
                hiddenWeights[i][j] -= hiddenDerivatives[current][i] * input[current][j] * learningRate * avg;
            }
            hiddenWeights[i][inputSize] -= hiddenDerivatives[current][i] * learningRate * avg;
        });
        logger.debug("single backward pass ended");
    }

    void teach(float[][] input, int realSize, int[] labels, float learningRate) {
        this.input = input;

        batchForwardPass(realSize);
        batchBackwardPass(labels, realSize, learningRate);
    }

    private void batchBackwardPass(int[] labels, int realSize, float learningRate) {
        logger.debug("started batch backward pass");
        for (int i = 0; i < realSize; i++) {
            backwardPass(i, labels[i],  learningRate);
        }
        logger.debug("batch backward pass ended");
    }

    /**
     * @param position # of element in batch that is used to store test input
     * @return most probable class determined by probabilities in output vector
     */
    private int mostProbableClass(int position) {
        logger.debug("started most probable class identification");
        float max = output[position][0];
        int value = 0;
        for (int i = 0; i < outputSize; i++) {
            if (output[position][i] > max) {
                max = output[position][i];
                value = i;            //predicted class
            }
        }
        logger.debug("class {} is most probable, it's probability {}", value, max);
        return value;
    }

    int predict(float[] input) {
        // we will use first batch element as testing input
        int current = 0;

        logger.debug("started testing");
        IntStream.range(0, hiddenSize).parallel().forEach(i -> {
            hidden[current][i] = 0;
            for (int j = 0; j < inputSize; j++) {
                hidden[current][i] += hiddenWeights[i][j] * input[j];
            }
            hidden[current][i] += hiddenWeights[i][inputSize]; // adding threshold

            // end of hidden calculation
            hidden[current][i] = (float) Math.tanh(hidden[current][i]);
        });

        float outputMax = 0;
        for (int i = 0; i < outputSize; i++) {
            output[current][i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                output[current][i] += outputWeights[i][j] * hidden[current][j];
            }
            output[current][i] += outputWeights[i][hiddenSize]; // adding threshold

            if (outputMax < output[current][i]) {
                outputMax = output[current][i];
            }
        }
        outputMax /= 2;

        for (int i = 0; i < outputSize; i++) {
            output[current][i] = Utils.softMax(output[current], outputMax, i);
        }

        int result = mostProbableClass(current);
        logger.debug("testing ended");
        return result;
    }
}
