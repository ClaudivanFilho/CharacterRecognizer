package com.classifier;

import static com.classifier.ClassifierSetBuilder.CAPACITY;
import static com.classifier.ClassifierSetBuilder.INDEX;
import java.io.File;
import java.io.Serializable;
import weka.classifiers.Classifier;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

import weka.core.Instance;

/**
 * Created by marcos on 4/17/16.
 */
public class MyClassifier implements Serializable {

    private final Classifier model;
    private final ClassifierSetBuilder setBuilder;
    private final FastVector classes;
    
    public String last_train;
    public String classifierName;
    
    public static String[] classifiers = {"MultilayerPerceptron", "BayesNet", "DecisionStump"};


    public MyClassifier(FastVector classes, String classifier) {
        this.classes = classes;
        if (classifier.equals("MultilayerPerceptron")) {
            this.model = new MultilayerPerceptron();
        } else if (classifier.equals("BayesNet")) {
            this.model = new BayesNet();
        } else if (classifier.equals("DecisionStump")) {
            this.model = new DecisionStump();
        } else {
            this.model = new MultilayerPerceptron();
        }
        this.classifierName = classifier;
        this.setBuilder = new ClassifierSetBuilder(classes);
        this.last_train = "";
    }

    public void buildClassifier() throws Exception {
        this.model.buildClassifier(getSet());
    }

    public Classifier getClassifier() {
        return this.model;
    }
    
    public String classifyInstance(String imagePath) throws Exception {    
        FastVector wekaAttributes = new FastVector(CAPACITY);
        for (int i = 0; i < INDEX; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAttributes.addElement(attr);
        }
        Attribute attr = new Attribute("classes", classes);
        wekaAttributes.addElement(attr);
        
        Instances trainingSet = new Instances("Rel", wekaAttributes, 1);
        trainingSet.setClassIndex(INDEX);
        
        File f = new File(imagePath);
        double[] histogram = Histogram.buildHistogram(f);
        Instance imageInstance = new Instance(ClassifierSetBuilder.CAPACITY);
        for (int i = 0; i < histogram.length; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i), histogram[i]);
        }
        String predicted = getSet().classAttribute().value((int) this.model.classifyInstance(imageInstance));
        return predicted;
    }

    public Instances getSet() {
        return this.setBuilder.getSet();
    }

    public void buildSet(String folderName, String classe) throws Exception {
        setBuilder.buildSet(folderName, classe);
    }
}
