package com.serializer;

import com.classifier.MyClassifier;
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

    public static void save(MyClassifier classifier) {
        try {
            FileOutputStream fileOut
                    = new FileOutputStream("classifier-" + classifier.classifierName + ".ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(classifier);
            out.close();
            fileOut.close();
            System.out.printf("Serialized data is saved in classifier-" + classifier.classifierName + ".ser");
        } catch (IOException i) {
            i.printStackTrace();
        }
    }
    
    public static boolean hasClassifier(String classifier) {
        File f = new File("classifier-" + classifier + ".ser");
        return f.exists() && !f.isDirectory();
    }

    public static MyClassifier load(String classifier) {
        MyClassifier n;
        try {
            FileInputStream fileIn = new FileInputStream("classifier-" + classifier + ".ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            n = (MyClassifier) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException i) {
            i.printStackTrace();
            return null;
        } catch (ClassNotFoundException c) {
            System.out.println("MyClassifier class not found");
            c.printStackTrace();
            return null;
        }
        return n;
    }

}