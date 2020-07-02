package ru.unn.itmm.fcnn.util;

import java.util.List;

public class MnistData {
    private List<float[]> trainImages;
    private int[] trainLabels;

    private List<float[]> testImages;
    private int[] testLabels;

    public List<float[]> getTrainImages() {
        return trainImages;
    }

    public void setTrainImages(List<float[]> trainImages) {
        this.trainImages = trainImages;
    }

    public int[] getTrainLabels() {
        return trainLabels;
    }

    public void setTrainLabels(int[] trainLabels) {
        this.trainLabels = trainLabels;
    }

    public List<float[]> getTestImages() {
        return testImages;
    }

    public void setTestImages(List<float[]> testImages) {
        this.testImages = testImages;
    }

    public int[] getTestLabels() {
        return testLabels;
    }

    public void setTestLabels(int[] testLabels) {
        this.testLabels = testLabels;
    }
}
