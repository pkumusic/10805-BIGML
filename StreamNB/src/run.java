import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * Created by MusicLee on 9/11/16.
 */
public class run extends Configured implements Tool {

    public static void main(String args[]) throws Exception {
        int res = ToolRunner.run(new run(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception{
        Path inputPath = new Path(args[0]);
        Path outputPath = new Path(args[1]);
        int num_reducers = Integer.valueOf(args[2]);

        Configuration conf = getConf();
        Job job = new Job(conf, "NB_train_Music");
        FileSystem fs = FileSystem.get(conf);

        if (fs.exists(outputPath))
            fs.delete(outputPath, true);

        job.setMapperClass(NB_train_hadoop.Map.class);
        job.setReducerClass(NB_train_hadoop.Reduce.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);

        job.setNumReduceTasks(num_reducers);

        job.setJarByClass(NB_train_hadoop.class);
        job.submit();

        return job.waitForCompletion(true) ? 0 : 1;
    }

}
