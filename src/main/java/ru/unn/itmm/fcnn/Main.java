package ru.unn.itmm.fcnn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.unn.itmm.fcnn.util.MnistData;
import ru.unn.itmm.fcnn.util.MnistReader;

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

        logger.info("train count: " + mnistData.getTrainImages().size());
        logger.info("test count: " + mnistData.getTestImages().size());
        logger.info("data is loaded");

        //configuring network
        Network network = new Network(batchSize, IMAGE_SIZE, hiddenSize, OUTPUT_SIZE);

        //for shuffle procedure
        int[] indexes = IntStream.range(0, mnistData.getTrainLabels().length).toArray();
        logger.info("start train");

        for (int epoch = 0; epoch < epochCount; epoch++) {
            logger.info("epoch: #" + epoch);
            shuffleArray(indexes);

            for (int i = 0; i < indexes.length / batchSize; i++) {
                float[][] input = new float[batchSize][];
                int[] labels = new int[batchSize];
                for (int j = 0; j < batchSize; j++) {
                    input[j] = mnistData.getTrainImages().get(indexes[i*batchSize + j]);
                    labels[j] = mnistData.getTrainLabels()[indexes[i*batchSize + j]];
                }
                network.teach(input, batchSize, labels, learningRate);
            }
        }
        logger.info("testing is started");
        int correctCount = 0;
        int actual, expected;
        for (int i = 0; i < mnistData.getTestLabels().length; i++) {
            actual = network.test(mnistData.getTestImages().get(i));
            expected = mnistData.getTestLabels()[i];
            if (actual == expected)
                correctCount++;
        }

        System.out.println("---- Result accuracy: " + (float) correctCount / mnistData.getTestLabels().length * 100);
    }
}
