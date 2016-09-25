import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;

/**
 * Created by MusicLee on 9/11/16.
 */
public class run{
    public static void main(String[] args) throws Exception {
        Path inputPath = new Path(args[0]);
        Path outputPath = new Path(args[1]);
        int num_reducers = Integer.valueOf(args[2]);

        JobConf conf = new JobConf(NB_train_hadoop.class);
        conf.setJobName("NBTRAIN");
//        FileSystem fs = FileSystem.get(conf);

//        if (fs.exists(outputPath))
//            fs.delete(outputPath, true);

        conf.setMapperClass(NB_train_hadoop.Map.class);
        conf.setReducerClass(NB_train_hadoop.Reduce.class);

        conf.setInputFormat(TextInputFormat.class);
        conf.setOutputFormat(TextOutputFormat.class);

        conf.setOutputKeyClass(Text.class);
        conf.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(conf, inputPath);
        FileOutputFormat.setOutputPath(conf, outputPath);
        conf.setNumReduceTasks(num_reducers);

        JobClient.runJob(conf);
        return;
        //job.submit();
    }

}
