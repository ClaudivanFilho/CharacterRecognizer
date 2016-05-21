package com.classifier;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by marcos on 4/17/16.
 */
public class ClassifierSetBuilder implements Serializable {
    
    public static final int CAPACITY = 257;
    public static final int INDEX = 256;
    
    private final Instances set;
    private final FastVector wekaAttributes;
    private final ArrayList<String> paths;

    public ClassifierSetBuilder(FastVector classes) {
        paths = new ArrayList<String>();
        FastVector wekaAttrs = new FastVector(CAPACITY);
        for (int i = 0; i < INDEX; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAttrs.addElement(attr);
        }
        Attribute attr = new Attribute("classes", classes);

        wekaAttrs.addElement(attr);
        Instances trainingSet = new Instances("Rel", wekaAttrs, 1);
        trainingSet.setClassIndex(INDEX);
        this.set = trainingSet;
        this.wekaAttributes = wekaAttrs;
    }

    public void buildSet(String folderName, String clazz) throws Exception {
        File folder = new File(folderName);
        if (folder.listFiles() == null) {
            folder.mkdir();
        }
        File[] listOfFiles = folder.listFiles();
        for (File f : listOfFiles) {
            if (paths != null) {
                paths.add(f.getPath());
            }
            double[] histogram = Histogram.buildHistogram(f);
            createSet(wekaAttributes, histogram, clazz);
        }
    }

    private void createSet(FastVector wekaAttributes, double[] histogram, String classe) {
        Instance imageInstance = new Instance(CAPACITY);
        for (int i = 0; i < histogram.length; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i), histogram[i]);
        }
        if (!classe.isEmpty()) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(INDEX), classe);
        }
        this.set.add(imageInstance);
    }

    public Instances getSet() {
        return this.set;
    }
    
    public List<String> getPaths() {
        return paths;
    }
}
