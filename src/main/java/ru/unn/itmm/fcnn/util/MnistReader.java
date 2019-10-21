package ru.unn.itmm.fcnn.util;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static java.lang.String.format;

public class MnistReader {
	public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
	public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;

	public static MnistData read(String trainImages, String trainLabels, String testImages, String testLabels) {
		MnistData data = new MnistData();

		data.setTrainImages(getImages(trainImages));
		data.setTrainLabels(getLabels(trainLabels));

		data.setTestImages(getImages(testImages));
		data.setTestLabels(getLabels(testLabels));

		return data;
	}

	public static int[] getLabels(String infile) {

		ByteBuffer bb = loadFileToByteBuffer(infile);

		assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];

		for (int i = 0; i < numLabels; ++i)
			labels[i] = bb.get() & 0xFF; // To unsigned

		return labels;
	}

	public static List<float[]> getImages(String infile) {
		ByteBuffer bb = loadFileToByteBuffer(infile);

		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

		int numImages = bb.getInt();
		int numRows = bb.getInt();
		int numColumns = bb.getInt();
		int res = numRows * numColumns;
		List<float[]> images = new ArrayList<>();

		for (int i = 0; i < numImages; i++)
			images.add(readImage(res, bb));

		return images;
	}

	private static float[] readImage(int res, ByteBuffer bb) {
		float[] row = new float[res];
		for (int pix = 0; pix < res; ++pix) {
			int current = bb.get() & 0xFF;
			row[pix] = current / 255.0f; // To unsigned
		}
		return row;
	}

	public static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
		if (expectedMagicNumber != magicNumber) {
			switch (expectedMagicNumber) {
			case LABEL_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not a label file.");
			case IMAGE_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not an image file.");
			default:
				throw new RuntimeException(
						format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
			}
		}
	}

	/*******
	 * Just very ugly utilities below here. Best not to subject yourself to
	 * them. ;-)
	 ******/
	private static String getFilePath(String fileName) {
	    return null;
	}

	public static ByteBuffer loadFileToByteBuffer(String infile) {
		return ByteBuffer.wrap(loadFile(infile));
	}

	public static byte[] loadFile(String fileName) {
		try (InputStream is = MnistReader.class.getClassLoader().getResource(fileName).openStream()){
		    byte[] bytes = new byte[is.available()];
		    is.read(bytes);
		    return bytes;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static String renderImage(int[][] image) {
		StringBuffer sb = new StringBuffer();

		for (int row = 0; row < image.length; row++) {
			sb.append("|");
			for (int col = 0; col < image[row].length; col++) {
				int pixelVal = image[row][col];
				if (pixelVal == 0)
					sb.append(" ");
				else if (pixelVal < 256 / 3)
					sb.append(".");
				else if (pixelVal < 2 * (256 / 3))
					sb.append("x");
				else
					sb.append("X");
			}
			sb.append("|\n");
		}

		return sb.toString();
	}

	public static String repeat(String s, int n) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++)
			sb.append(s);
		return sb.toString();
	}
}
