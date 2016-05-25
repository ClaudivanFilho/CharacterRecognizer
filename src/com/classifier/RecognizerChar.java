package com.classifier;

import weka.classifiers.Evaluation;
import weka.core.FastVector;
import weka.core.Instances;

public class RecognizerChar {

    public String PATH_TRAIN = "";
    public String PATH_TEST_SET = "";
    
    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";

    private final boolean verbose = Boolean.parseBoolean("-v");
    private final FastVector classes;
    private Evaluation eTest;
    private Instances testSet;
    private ClassifierSetBuilder testBuilder;
    
    public MyClassifier NNCTrain;

    public RecognizerChar() {
        classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);        
    }

    public void train(String classifier) throws Exception {
        NNCTrain = new MyClassifier(classes, classifier);
        NNCTrain.buildSet(PATH_TRAIN + "/digitos", DIGITOS);
        NNCTrain.buildSet(PATH_TRAIN + "/letras", LETRAS);
        NNCTrain.buildSet(PATH_TRAIN + "/digitos_letras", DIGITOS_LETRAS);
        NNCTrain.buildSet(PATH_TRAIN + "/sem_caracteres", SEM_CARACTERES);
        NNCTrain.buildClassifier();        
    }
    
    public String classsifyImage(String pathImage) throws Exception {
        return NNCTrain.classifyInstance(pathImage);
    }
    
    public String runSetTest() throws Exception {
        initTest();
        return getStatistics();
    }

    private void initTest() throws Exception {
        testBuilder = new ClassifierSetBuilder(classes);
        testBuilder.buildSet(PATH_TEST_SET + "/letras", LETRAS);
        testBuilder.buildSet(PATH_TEST_SET + "/digitos", DIGITOS);
        testBuilder.buildSet(PATH_TEST_SET + "/digitos_letras", DIGITOS_LETRAS);
        testBuilder.buildSet(PATH_TEST_SET + "/sem_caracteres", SEM_CARACTERES);
        testSet = testBuilder.getSet();
        eTest = new Evaluation(NNCTrain.getSet());
        eTest.evaluateModel(NNCTrain.getClassifier(), testSet);
    }
   
    private String getStatistics() throws Exception {
        String statistics = "";
        for (int i = 0; i < testSet.numInstances(); i++) {
            double pred = NNCTrain.getClassifier().classifyInstance(testSet.instance(i));
            statistics += "ID: " + testBuilder.getPaths().get(i) + "\n";
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
