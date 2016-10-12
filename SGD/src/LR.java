import java.io.*;
import java.util.*;

/**
 * Created by MusicLee on 10/6/16.
 */
public class LR {
    HashMap<String, int[]> label_A = new HashMap<String, int[]>();
    HashMap<String, double[]> label_B = new HashMap<String, double[]>();
    static double overflow=20;

    public void initialize(int N, String labels){
        String[] ls = labels.split(" ");
        for (String l:ls){
            label_A.put(l, new int[N+1]);
            label_B.put(l, new double[N+1]);
        }
        return;
    }

    public void train(BufferedReader in, int N, float lambda, float mu, int max_iter, int train_size) throws IOException {
        String line;
        int k = 0;
        float init_lambda = lambda;
        int total_size = max_iter * train_size;
        int epoch = 0;
        while ((line = in.readLine()) != null) {
            k += 1;
            if(k > total_size) break;
            if(k % train_size == 1) epoch += 1;
            lambda = init_lambda / (epoch * epoch);
            String[] infos = line.split("\t");
            String[] labels = infos[1].split(",");
            String doc = infos[2];
            Vector<String> tokens = tokenizeDoc(doc);
            HashMap<Integer, Integer> x = new HashMap<Integer, Integer>();
            for (String token: tokens){
                int hash = word2hash(N, token);
                if(x.containsKey(hash)){continue;}
                else{x.put(hash, 1);}
            }
            x.put(N, 1); // add bias
            for (String label: label_A.keySet()) {
                // for each label
                // if label in labels, y=1; else y=0
                int[] A = label_A.get(label);
                double[] B = label_B.get(label);
                int y = 0;
                for (String l : labels) {
                    if (l.equals(label)) y = 1;
                }
                double accum = 0;
                for (int hash : x.keySet()) {
                    accum += B[hash] * x.get(hash);
                }
                double p = sigmoid(accum);
                for (int hash : x.keySet()) {
                    if(hash == N){
                        B[hash] += lambda * (y - p) * x.get(hash); //update bias
                        continue;
                    }
                    B[hash] *= Math.pow((double) (1 - lambda * mu), (double) (k - A[hash]));
                    B[hash] += lambda * (y - p) * x.get(hash);
                    A[hash] = k;
                }
            }
        }
        // Update at the end
        for (String label: label_A.keySet()){
            int[] A = label_A.get(label);
            double[] B = label_B.get(label);
            for (int i=0; i<N; i++){
                B[i] *= Math.pow((double) (1 - lambda * mu), (double) (k - A[i]));
                A[i] = k;
            }
        }

        //for (int i=0;i<N;i++)
        //System.out.print(label_B.get("Location")[i]);
    }

    double sigmoid(double score){
        if (score > overflow) score = overflow;
        else if (score < -overflow) score = -overflow;
        double exp = Math.exp(score);
        return exp/(1+exp);
    }

    public int word2hash(int N, String word){
        int id = word.hashCode() % N;
        if (id < 0) id+= N;
        return id;
    }

    public Vector<String> tokenizeDoc(String cur_doc){
        String[] words = cur_doc.split("\\s+");
        Vector<String> tokens = new Vector<String>();
        for (int i=0; i<words.length; i++){
            words[i] = words[i].replaceAll("\\W", "");
            if (words[i].length() > 0){
                tokens.add(words[i]);
            }
        }
        return tokens;
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

    void test(String test_file, int N) throws IOException {
        File fin = new File(test_file);
        FileInputStream fis = new FileInputStream(fin);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        while ((line = br.readLine()) != null) {
            String[] infos = line.split("\t");
            String[] labels = infos[1].split(",");
            String doc = infos[2];
            Vector<String> tokens = tokenizeDoc(doc);
            HashMap<Integer, Integer> x = new HashMap<Integer, Integer>();
            for (String token: tokens){
                int hash = word2hash(N, token);
                if(x.containsKey(hash)){x.put(hash, x.get(hash)+1);}
                else{x.put(hash, 1);}
            }
            x.put(N, 1); // add bias
            int count = 0;
//            for (String label:labels){
//                System.out.print(label + ",");
//            }
//            System.out.print(": ");
            for (String label: label_A.keySet()) {
                double[] B = label_B.get(label);
                double accum = 0;
                for (int hash : x.keySet()) {
                    accum += B[hash] * x.get(hash);
                }
                double p = sigmoid(accum);
//                if (p>0.5){
                if (count == 0) System.out.print(label + "\t" + new Double(p).toString());
                else System.out.print(',' + label + "\t" + new Double(p).toString());
                count += 1;
//                }
            }
            System.out.println();
        }
    }



    public static void main(String[] args) throws Exception{
        // Prints "Hello, World" to the terminal window.
        //System.out.println("Program started");
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
        lr.initialize(N, labels);
        lr.train(in, N, lambda, mu, max_iter, train_size);
        lr.test(test_file, N);

    }

    public void print(int i){
        System.out.println(i);
    }
    public void print(String s){
        System.out.println(s);
    }
    public void print(float s){
        System.out.println(s);
    }
    public void print(Set<String> set){
        for (String s : set){
            System.out.print(" "+s);
        }
    }
}
