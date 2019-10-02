package ru.unn.itmm.fcnn.util;

import java.util.concurrent.ThreadLocalRandom;

public abstract class Utils {

    public static float random() {
        return (ThreadLocalRandom.current().nextFloat() - 0.5f) * 0.2f;
    }

    public static void initRandomWeights(float[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = random();
            }
        }
    }

    public static float softMax(float[] input, float threshold, int current) {
        float sumExp = 0;
        for (int i = 0; i < input.length; i++) {
            sumExp += (float) Math.exp(input[i] - threshold);
        }
        float result =  (float) Math.exp(input[current] - threshold) / sumExp;
        return result;
    }

    public static float tanhDerivative(float value) {
        return 1 - value*value;
    }
}
