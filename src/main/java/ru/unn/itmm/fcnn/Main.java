package ru.unn.itmm.fcnn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.unn.itmm.fcnn.util.MnistData;
import ru.unn.itmm.fcnn.util.MnistReader;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    private static final int IMAGE_SIZE = 28 * 28;
    private static final int OUTPUT_SIZE = 10;

    // Implementing Fisher–Yates shuffle
    static void shuffleArray(int[] ar)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.out.println("Not enough arguments are passed. Format:");
            System.out.println("[learning_rate] [epoch] [hidden layer size] [batch size]");
            return;
        }

        float learningRate = Float.parseFloat(args[0]);
        int epochCount = Integer.parseInt(args[1]);
        int hiddenSize = Integer.parseInt(args[2]);
        int batchSize = Integer.parseInt(args[3]);

        //load dataset
        MnistData mnistData = MnistReader.read(
                "train-images.idx3-ubyte",
                "train-labels.idx1-ubyte",
                "t10k-images.idx3-ubyte",
                "t10k-labels.idx1-ubyte"
        );

        logger.info("MNIST dataset is loaded");
        logger.info("Training dataset size: {}", mnistData.getTrainImages().size());
        logger.info("Test dataset size: {}", mnistData.getTestImages().size());

        //configuring network
        Network network = new Network(batchSize, IMAGE_SIZE, hiddenSize, OUTPUT_SIZE);

        //for shuffle procedure
        int[] indexes = IntStream.range(0, mnistData.getTrainLabels().length).toArray();
        logger.info("Training is started");
        long startTime = System.nanoTime();

        for (int epoch = 0; epoch < epochCount; epoch++) {
            logger.info("Epoch: #{}...", epoch);
            shuffleArray(indexes);

            for (int i = 0; i < indexes.length / batchSize; i++) {
                float[][] input = new float[batchSize][];
                int[] labels = new int[batchSize];

                int startIndex = i*batchSize;
                int endIndex = (i+1)*batchSize;
                if (endIndex > indexes.length) {
                    endIndex = indexes.length;
                }

                for (int j = startIndex; j < endIndex; j++) {
                    input[j-startIndex] = mnistData.getTrainImages().get(indexes[j]);
                    labels[j-startIndex] = mnistData.getTrainLabels()[indexes[j]];
                }
                network.teach(input, endIndex-startIndex, labels, learningRate);
            }
        }
        long endTime = System.nanoTime();
        long secondsSpent = (endTime - startTime) / 1_000_000_000;

        logger.info("Training is ended, time spent: {} sec.", secondsSpent);

        logger.info("Train accuracy: {}", test(network, mnistData.getTrainImages(), mnistData.getTrainLabels()));

        logger.info("Testing is started");
        logger.info("Test accuracy: {}", test(network, mnistData.getTestImages(), mnistData.getTestLabels()));
    }

    private static float test(Network network, List<float[]> images, int[] labels) {
        int correctCount = 0;
        int actual, expected;
        for (int i = 0; i < labels.length; i++) {
            actual = network.predict(images.get(i));
            expected = labels[i];
            if (actual == expected)
                correctCount++;
        }
        return (float) correctCount / labels.length * 100;
    }
}
