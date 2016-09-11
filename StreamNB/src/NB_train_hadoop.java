import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Vector;

/**
 * Created by MusicLee on 9/11/16.
 */
public class NB_train_hadoop {

    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        String filename;
        HashMap<String, Integer> counters;
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


        public void update_count(String key){
            if(counters.containsKey(key)){
                counters.put(key, counters.get(key)+1);
            }
            else{
                counters.put(key, 1);
            }
            return;
        }

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            counters = new HashMap<String, Integer>();
            FileSplit fsFileSplit = (FileSplit) context.getInputSplit();
            filename = context.getConfiguration().get(fsFileSplit.getPath().getParent().getName());
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            //Path pt=new Path(filename);//Location of file in HDFS
            //FileSystem fs = FileSystem.get(new Configuration());
            //BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
            BufferedReader br = new BufferedReader(new StringReader(value.toString()));
            String line = null;
            while ((line = br.readLine()) != null) {
                //System.out.println(line);
                String[] splitted = line.split("\t");
                String[] labels = splitted[0].split(",");
                String cur_doc= splitted[1];
                Vector<String> tokens = tokenizeDoc(cur_doc);
                for (String label : labels) {
//                    update_count("Y=" + label);
//                    update_count("Y=*");
                    context.write(new Text("Y="+label), one);
                    context.write(new Text("Y=*"), one);
                    for (String token : tokens) {
//                        update_count("Y=" + label + ",W=*");
//                        update_count("Y=" + label + ",W=" + token);
                        context.write(new Text("Y="+label+",W=*"), one);
                        context.write(new Text("Y="+label+",W="+token), one);
                    }

                }
            }
            br.close();

//            for (java.util.Map.Entry<String, Integer> entry : counters.entrySet()) {
//                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
//            }

        }
    }

    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
            int sum = 0;
            for (IntWritable val: values){
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

}
