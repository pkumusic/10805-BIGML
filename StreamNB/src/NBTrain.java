// Author: Music
import java.io.*;
import java.util.*;

public class NBTrain{

    HashMap<String, Integer> counters;

    public NBTrain(){
        counters = new HashMap<String, Integer>();
    }

    public void train (String filePath) throws Exception{
        File fin = new File(filePath);
        FileInputStream fis = new FileInputStream(fin);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        //HashMap<String, Integer> counters = new HashMap<String, Integer>();
        while ((line = br.readLine()) != null) {
            //System.out.println(line);
            String[] splitted = line.split("\t");
            String[] labels = splitted[0].split(",");
            String cur_doc= splitted[1];
            Vector<String> tokens = tokenizeDoc(cur_doc);
            for (String label : labels) {
                if (label.endsWith("CAT")) {
                    update_count("Y=" + label);
                    update_count("Y=*");
                    for (String token : tokens) {
                        update_count("Y=" + label + ",W=*");
                        update_count("Y=" + label + ",W=" + token);
                    }
                }
            }
        }
        br.close();
        printMap(counters);
    }

    public void printMap(Map<String, Integer> map){
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + '\t' + entry.getValue());
        }
    }

    public void update_count(String key){
        if(counters.containsKey(key)){
            counters.put(key, counters.get(key)+1);
        }
        else{
            counters.put(key, 1);
        }
        return;
    }


    static Vector<String> tokenizeDoc(String cur_doc){
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



    public static void main(String[] args) throws Exception{
        // Prints "Hello, World" to the terminal window.
        System.out.println("Program started");
        NBTrain nbTrain =new NBTrain();
        // Load data
        // System.out.println(new File(".").getCanonicalPath()); // Find the root path
        //String train_pjath = "StreamNB/RCV1/RCV1.small_train.txt";
        String train_path = args[0];
        nbTrain.train(train_path);

    }
}