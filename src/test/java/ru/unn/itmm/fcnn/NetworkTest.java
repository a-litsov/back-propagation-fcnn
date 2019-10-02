package ru.unn.itmm.fcnn;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.unn.itmm.fcnn.util.MnistData;
import ru.unn.itmm.fcnn.util.MnistReader;

import java.util.List;
import java.util.stream.IntStream;

import static ru.unn.itmm.fcnn.Main.shuffleArray;

@RunWith(JUnit4.class)
public class NetworkTest {
    private static final Logger logger = LoggerFactory.getLogger(NetworkTest.class);

    private static final MnistData mnistData = MnistReader.read(
            "train-images.idx3-ubyte",
            "train-labels.idx1-ubyte",
            "t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte"
    );

    @Test
    public void testBigLearningRate() {
        float accuracy = test(0.1f, 10, 20);
        Assert.assertTrue("Accuracy:" + accuracy,accuracy >= 87);
    }

    @Test
    public void testLowLearningRateLowHiddenLayerNeuronsCount() {
        float accuracy = test(0.01f, 10, 20);
        Assert.assertTrue("Accuracy:" + accuracy,accuracy >= 92);
    }

    @Test
    public void testLowLearningRateModerateHiddenLayerNeuronsCount() {
        float accuracy = test(0.05f, 10, 50);
        Assert.assertTrue("Accuracy:" + accuracy,accuracy >= 92);
    }

    @Test
    public void testLowLearningRateBigHiddenLayerNeuronsCount() {
        float accuracy = test(0.05f, 10, 80);
        Assert.assertTrue("Accuracy:" + accuracy, accuracy >= 92);
    }

    private static final float test(float learningRate, int epochCount, int hiddenSize) {
        int imageSize = 28 * 28;
        int batchSize = 250;
        int outputSize = 10;

        //configuring network
        Network network = new Network(batchSize, imageSize, hiddenSize, outputSize);

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
                    input[j] = mnistData.getTrainImages().get(indexes[i * batchSize + j]);
                    labels[j] = mnistData.getTrainLabels()[indexes[i * batchSize + j]];
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
        return (float) correctCount / mnistData.getTestLabels().length * 100;
    }
}
