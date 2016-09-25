import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.Iterator;
import java.util.Vector;

/**
 * Created by MusicLee on 9/11/16.
 */
public class NB_train_hadoop {

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable> {
        //String filename;
        //HashMap<String, Integer> counters;
        private final static IntWritable one = new IntWritable(1);
        //private Text word = new Text();

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


//        public void update_count(String key){
//            if(counters.containsKey(key)){
//                counters.put(key, counters.get(key)+1);
//            }
//            else{
//                counters.put(key, 1);
//            }
//            return;
//        }        public void update_count(String key){
//            if(counters.containsKey(key)){
//                counters.put(key, counters.get(key)+1);
//            }
//            else{
//                counters.put(key, 1);
//            }
//            return;
//        }

//        @Override
//        protected void setup(Context context) throws IOException, InterruptedException {
//            counters = new HashMap<String, Integer>();
//            FileSplit fsFileSplit = (FileSplit) context.getInputSplit();
//            filename = context.getConfiguration().get(fsFileSplit.getPath().getParent().getName());
//        }

        @Override
        public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> context, Reporter reporter) throws IOException{
            //Path pt=new Path(filename);//Location of file in HDFS
            //FileSystem fs = FileSystem.get(new Configuration());
            //BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
            BufferedReader br = new BufferedReader(new StringReader(value.toString()));
            String line = null;
            while ((line = br.readLine()) != null) {
                //System.out.println(line);
                String[] splitted = line.split("\t");
                String[] labels = splitted[1].split(",");
                String cur_doc= splitted[2];
                Vector<String> tokens = tokenizeDoc(cur_doc);
                for (String label : labels) {
//                    update_count("Y=" + label);
//                    update_count("Y=*");
                    context.collect(new Text("Y="+label), one);
                    for (String token : tokens) {
//                        update_count("Y=" + label + ",W=*");
//                        update_count("Y=" + label + ",W=" + token);
                        context.collect(new Text("Y="+label+",W="+token), one);
                    }
                    context.collect(new Text("Y="+label+",W=*"), new IntWritable(tokens.size()));
                }
                context.collect(new Text("Y=*"), new IntWritable(labels.length));
            }
            br.close();

//            for (java.util.Map.Entry<String, Integer> entry : counters.entrySet()) {
//                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
//            }

        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text, IntWritable> context, Reporter reporter) throws IOException{
            int sum = 0;
            while (values.hasNext()){
                sum += values.next().get();
            }
            context.collect(key, new IntWritable(sum));
        }
    }

}
