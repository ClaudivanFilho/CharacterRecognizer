package com.serializer;

import com.classifier.NeuralNetworkClassifier;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 *
 * @author claudivan
 */
public class ClassifierSerializer {

    public static void save(NeuralNetworkClassifier classifier) {
        try {
            FileOutputStream fileOut
                    = new FileOutputStream("classifier.ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(classifier);
            out.close();
            fileOut.close();
            System.out.printf("Serialized data is saved in classifier.ser");
        } catch (IOException i) {
            i.printStackTrace();
        }
    }
    
    public static boolean hasClassifier() {
        File f = new File("classifier.ser");
        return f.exists() && !f.isDirectory();
    }

    public static NeuralNetworkClassifier load() {
        NeuralNetworkClassifier n;
        try {
            FileInputStream fileIn = new FileInputStream("classifier.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            n = (NeuralNetworkClassifier) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException i) {
            i.printStackTrace();
            return null;
        } catch (ClassNotFoundException c) {
            System.out.println("NeuralNetworkClassifier class not found");
            c.printStackTrace();
            return null;
        }
        return n;
    }

}
