package com.classifier;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

/**
 * Created by claudivan on 4/17/16.
 */
public class Histogram {

    private static final double LUMINANCE_RED = 0.333333333333;
    private static final double LUMINANCE_GREEN = 0.33333333;
    private static final double LUMINANCE_BLUE = 0.333333333333333;
    private static final int HIST_WIDTH = 256;
    private static final int HIST_HEIGHT = 100;
    private static final int THRESHOLD = 128;
    
    /**
     * Parses pixels out of an image file, converts the RGB values to
     * its equivalent grayscale value (0-255), then constructs a
     * histogram of the percentage of counts of grayscale values.
     *
     * @param infile - the image file.
     * @return - a histogram of grayscale percentage counts.
     * @throws java.lang.Exception
     */
    protected static double[] buildHistogram(File infile) throws Exception {
        BufferedImage input = ImageIO.read(infile);
        input = resizeImage(input);
        int width = input.getWidth();
        int height = input.getHeight();
        List<Integer> graylevels = new ArrayList<Integer>();
        for (int col = 0; col < height; col++) {
            for (int row = 0; row < width; row++) {
                Color c = new Color(input.getRGB(row, col));
                int graylevel = (int) (LUMINANCE_RED * c.getRed() +
                        LUMINANCE_GREEN * c.getGreen() +
                        LUMINANCE_BLUE * c.getBlue());
                if (graylevel > THRESHOLD) {
                    graylevels.add(col);
                }
            }
        }
        double[] histogram = new double[HIST_WIDTH];
        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
            int idx = graylevel;
            histogram[idx] +=
                    Collections.frequency(graylevels, graylevel);
        }
        return histogram;
    }
    
    private static BufferedImage resizeImage(BufferedImage originalImage){
        int type = originalImage.getType() == 0? BufferedImage.TYPE_INT_ARGB : originalImage.getType();
	BufferedImage resizedImage = new BufferedImage(HIST_WIDTH, HIST_WIDTH, type);
	Graphics2D g = resizedImage.createGraphics();
	g.drawImage(originalImage, 0, 0, HIST_WIDTH, HIST_WIDTH, null);
	g.dispose();
		
	return resizedImage;
    }
}
