package com.classifier;

import weka.classifiers.Evaluation;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class RecognizerChar {

    public static String PATH_TRAIN = "";
    public static String PATH_TEST_SET = "";
    
    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";

    static List<String> paths = new ArrayList<String>();
    static NeuralNetworkClassifier NNCTrain;
    static Evaluation eTest;
    static Instances testSet;

    static boolean verbose;
    static FastVector classes;

    /**
     *
     * @throws Exception
     */
    public static void train() throws Exception {
        verbose = Boolean.parseBoolean("-v");

        classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);

        NNCTrain = new NeuralNetworkClassifier(classes);
        String folderDigits = PATH_TRAIN + "/digitos";
        String folderLetters = PATH_TRAIN + "/letras";
        String folderBoth = PATH_TRAIN + "/digitos_letras";
        String nothing = PATH_TRAIN + "/sem_caracteres";

        NNCTrain.buildSet(folderDigits, DIGITOS, null);
        NNCTrain.buildSet(folderLetters, LETRAS, null);
        NNCTrain.buildSet(folderBoth, DIGITOS_LETRAS, null);
        NNCTrain.buildSet(nothing, SEM_CARACTERES, null);
        NNCTrain.buildClassifier();        
    }
    
    public static String classsifyImage(String pathImage) throws Exception {
        return NNCTrain.classifyInstance(pathImage);
    }
    
    public static String runSetTest() throws Exception {
        initTest(PATH_TEST_SET,classes);
        return printStatistics();
    }

    static private void initTest(String pathTest, FastVector classes) throws Exception {
        eTest = new Evaluation(NNCTrain.getSet());
        ClassifierSetBuilder testBuilder = new ClassifierSetBuilder(classes);
        String folderTestLetters = pathTest + "/letras";
        String folderTestDigits = pathTest + "/digitos";
        String folderTestBoth = pathTest + "/digitos_letras";
        String nothingTest = pathTest + "/sem_caracteres";

        testBuilder.buildSet(folderTestLetters, LETRAS, paths);
        testBuilder.buildSet(folderTestDigits, DIGITOS, paths);
        testBuilder.buildSet(folderTestBoth, DIGITOS_LETRAS, paths);
        testBuilder.buildSet(nothingTest, SEM_CARACTERES, paths);

        testSet = testBuilder.getSet();
        eTest.evaluateModel(NNCTrain.getClassifier(), testSet);
    }
   
    private static String printStatistics() throws Exception {
        String statistics = "";
        for (int i = 0; i < testSet.numInstances(); i++) {
            double pred = NNCTrain.getClassifier().classifyInstance(testSet.instance(i));
            statistics += "ID: " + paths.get(i) + "\n";
            String actual = testSet.classAttribute().value((int) testSet.instance(i).classValue());
            String predicted = testSet.classAttribute().value((int) pred);
            statistics += "actual: " + actual + "\n";
            statistics += "predicted: " + predicted + "\n \n";
            if (actual.equals(predicted)) {
                statistics += "SUCCESS" + "\n";
            } else {
                statistics += "FAILURE" + "\n";
            }
        }

        if (verbose) {
            statistics += eTest.toSummaryString(true) + "\n";
            statistics += eTest.toClassDetailsString() + "\n";
        }

        statistics += "precision: " + eTest.weightedPrecision() + "\n";
        statistics += "recall: " + eTest.weightedRecall() + "\n";
        statistics += "f-measure: " + eTest.weightedFMeasure() + "\n";
        return statistics;
    }
}
