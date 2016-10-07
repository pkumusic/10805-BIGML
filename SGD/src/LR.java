import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by MusicLee on 10/6/16.
 */
public class LR {
    HashMap<String, HashMap<Integer, Integer>> label_A;
    HashMap<String, HashMap<Integer, Float>> label_B;

    public void initialize(String labels){
        
    }

    public void train(BufferedReader in, int N, float lambda, float mu, int max_iter, int train_size) throws IOException {

        String line;
        while ((line = in.readLine()) != null) {
            // TODO: Handle input line
            System.out.println(line);
            break;
        }
    }

    public HashSet<String> readLabels(BufferedReader in) throws IOException {
        String line;
        HashSet<String> label_set = new HashSet<>();
        while ((line = in.readLine()) != null) {
            //System.out.println(line);
            String[] infos = line.split("\t");
            String[] labels = infos[1].split(",");
            String doc = infos[2];
            for (String label: labels){
                label_set.add(label);
            }
        }
        print(label_set);
        print(label_set.size());
        return label_set;
    }



    public static void main(String[] args) throws Exception{
        // Prints "Hello, World" to the terminal window.
        System.out.println("Program started");
        // dictionary_size:N, learning_rate:lambda, regularization:mu, max_iter, train_size, test_file
        if(args.length != 6){
            System.out.println(args.length);
            System.out.println("E.g. java LR dictionary_size learning_rate regulization_term max_iter train_size test_file");
            System.exit(0);
        }

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

        int N            =  Integer.parseInt(args[0]);
        float lambda         =  Float.parseFloat(args[1]);
        float mu     =  Float.parseFloat(args[2]);
        int max_iter     =  Integer.parseInt(args[3]);
        int train_size   =  Integer.parseInt(args[4]);
        String test_file =  args[5];
        //HashSet<String> label_set = lr.readLabels(in);
        LR lr =new LR();
        String labels = "Agent other Organisation TimePeriod Device Activity ChemicalSubstance MeanOfTransportation SportsSeason Biomolecule Work CelestialBody Event Person Species Place Location";
        lr.initialize(labels);
        //lr.train(in, N, lambda, mu, max_iter, train_size);
        //lr.test(test_file)
    }

    public void print(int i){
        System.out.println(i);
    }
    public void print(String s){
        System.out.println(s);
    }
    public void print(Set<String> set){
        for (String s : set){
            System.out.print(" "+s);
        }
    }
}
