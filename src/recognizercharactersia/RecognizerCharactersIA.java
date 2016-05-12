/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package recognizercharactersia;

import java.awt.Dimension;
import java.awt.Toolkit;
import javax.swing.UnsupportedLookAndFeelException;

/**
 *
 * @author claudivan
 */
public class RecognizerCharactersIA {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            javax.swing.UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
        } catch (ClassNotFoundException ex) {
        } catch (InstantiationException ex) {
        } catch (IllegalAccessException ex) {
        } catch (UnsupportedLookAndFeelException ex) {
        }

        TelaInicial frame = new TelaInicial();

        Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
        frame.setLocation(dim.width / 2 - frame.getSize().width / 2, dim.height / 2 - frame.getSize().height / 2);

        frame.setTitle("Reconhecedor de Caracteres");
        frame.setVisible(true);
    }

}
