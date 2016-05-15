package com.classifier;

import static com.classifier.ClassifierSetBuilder.CAPACITY;
import static com.classifier.ClassifierSetBuilder.INDEX;
import java.io.File;
import java.io.Serializable;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.List;
import weka.core.Instance;

/**
 * Created by marcos on 4/17/16.
 */
public class NeuralNetworkClassifier implements Serializable {

    private Classifier model;
    private ClassifierSetBuilder setBuilder;
    private FastVector classes;
    
    public String last_train;


    public NeuralNetworkClassifier(FastVector classes) {
        this.classes = classes;
        this.model = new MultilayerPerceptron();
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

    public void buildSet(String folderName, String classe, List<String> files) throws Exception {
        setBuilder.buildSet(folderName, classe, files);
    }
}
