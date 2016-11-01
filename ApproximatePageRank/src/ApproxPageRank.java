/**
 * Compute page ranks and subsample a graph of local community given a seed.
 *
 * Input
 *   argv[0]: path to adj graph. Each line has the following format
 *            <src>\t<dst1>\t<dst2>...
 *   argv[1]: seed node
 *   argv[2]: alpha
 *   argv[3]: epsilon
 *
 * Output
 *   Print to stdout. Lines have the following format
 *     <v1>\t<pagerank1>\n
 *     <vr>\t<pagerank2>\n
 *     ...
 *   Order does NOT matter.
 */

import com.sun.prism.shader.Solid_TextureSecondPassLCD_Loader;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;


public class ApproxPageRank {
    String path;
    String seed;
    float alpha;
    float epsilon;
    HashMap<String, Float> p = new HashMap<>();
    HashMap<String, Float> r = new HashMap<>();
    HashMap<String, Set<String>> sub_graph= new HashMap<>();

    ApproxPageRank(String path, String seed, float alpha, float epsilon){
        this.path = path;
        this.seed = seed;
        this.alpha= alpha;
        this.epsilon = epsilon;
        r.put(seed, new Float(1.0));
    }

    public void pageRank() throws IOException{
        // Go through the file, update r, p.
        int i = 0;
        while(true){
            boolean stop = true;
            i += 1;
            //print(i);
            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;
            while ((line = br.readLine()) != null) {
                String[] nodes = line.split("\t");
                String node = nodes[0];
                int degree = nodes.length - 1;
                if(degree == 0) continue;
                if(r.containsKey(node) && (r.get(node)/degree > epsilon) ){
                    push(node, nodes, degree);
                    stop = false;
                }
                else continue;
            }
            if(stop) break;
        }
        return;
    }

    public void push(String node, String[] nodes, int degree){
        // p(u) = p(u) + alpha * r(u)
        float r_u = r.get(node);
        if(p.containsKey(node)){p.put(node, p.get(node)+alpha*r_u);}
        else{p.put(node, alpha * r_u);}
        // r(u) = (1-alpha) * r(u) / 2
        r.put(node, r_u * (1-alpha) / 2);
        // r(v) = r(v) + (1-alpha) * r(u) / (2 * d(u))
        for(int i = 1; i < nodes.length; i++){
            String neighbor = nodes[i];
            if(r.containsKey(neighbor)){r.put(neighbor, r.get(neighbor) + (1-alpha)*r_u/(2 * degree));}
            else{r.put(neighbor, (1-alpha)*r_u/(2 * degree));}
        }
    }

    public void cacheSubGraph() throws IOException{
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            String[] nodes = line.split("\t");
            String node = nodes[0];
            int degree = nodes.length - 1;
            if (degree == 0) continue;
            if (p.containsKey(node)) {
                // add this node and its neighbors to the cached sub_graph
                //System.out.println(node);
                //System.out.println(p.get(node));
                Set<String> neighbors = new HashSet<>();
                for (int i = 1; i < nodes.length; i++) {
                    neighbors.add(nodes[i]);
                }
                sub_graph.put(node, neighbors);
            }
        }
    }

    public Set<String> findLowConductanceSubGraph(){
        Map<String, Float> sorted_p = sortByValues(p);
        Set<String> set = sorted_p.keySet();
        Iterator<String> iterator = set.iterator();

        HashSet<String> S = new HashSet();

        // add the node and re-calculate boundary and volume
        S.add(seed);
        float boundary = 0;
        float volume   = 0;
        Set<String> neighbors = sub_graph.get(seed);
        for(String neighbor: neighbors){
            if(!S.contains(neighbor)) boundary += 1;
        }

        HashSet<String> S_star = new HashSet(S);
        float min_phi = Float.POSITIVE_INFINITY;
        float phi;

        while(iterator.hasNext()) {
            String node = iterator.next();
            if(node.equals(seed)) continue;
            // add the node and re-calculate boundary and volume
            S.add(node);
            int in_S = 0;
            int out_S = 0;
            neighbors = sub_graph.get(node);
            for(String neighbor: neighbors){
                if(S.contains(neighbor)) in_S ++;
                else out_S ++;
            }
            boundary += out_S - in_S;
            volume   += in_S;
            phi = boundary / volume;
            if(phi < min_phi){
                min_phi = phi;
                S_star = new HashSet(S);
            }
            //System.out.println(min_phi);
            //System.out.println(me.getValue());
        }
        return S_star;
    }

    public void output(Set<String> lc){
        for (String node: lc){
            System.out.println(node + "\t" + p.get(node).toString());
        }
    }

    public static void main(String[] args) throws IOException {
        // TODO: start your code here
        if (args.length != 4) {
            System.out.println(args.length);
            System.out.println("E.g. java ApproxPageRank input-path seed alpha epsilon");
            System.exit(0);
        }

        String path = args[0];
        String seed = args[1];
        float alpha = Float.parseFloat(args[2]);
        float epsilon = Float.parseFloat(args[3]);
        ApproxPageRank approxPageRank = new ApproxPageRank(path, seed, alpha, epsilon);
        approxPageRank.pageRank();
        approxPageRank.cacheSubGraph();
        Set<String> lc = approxPageRank.findLowConductanceSubGraph();
        approxPageRank.output(lc);
    }

    public HashMap sortByValues(HashMap map) {
        List list = new LinkedList(map.entrySet());
        // Defined Custom Comparator here
        Collections.sort(list, new Comparator() {
            public int compare(Object o1, Object o2) {
                return ((Comparable) ((Map.Entry) (o2)).getValue())
                        .compareTo(((Map.Entry) (o1)).getValue());
            }
        });
        // Here I am copying the sorted list in HashMap
        // using LinkedHashMap to preserve the insertion order
        HashMap sortedHashMap = new LinkedHashMap();
        for (Iterator it = list.iterator(); it.hasNext();) {
            Map.Entry entry = (Map.Entry) it.next();
            sortedHashMap.put(entry.getKey(), entry.getValue());
        }
        return sortedHashMap;
    }


    public void print(int var1) {
        System.out.println(var1);
    }

    public void print(String var1) {
        System.out.println(var1);
    }

    public void print(float var1) {
        System.out.println(var1);
    }

    public void print(Set<String> var1) {
        Iterator var2 = var1.iterator();

        while (var2.hasNext()) {
            String var3 = (String) var2.next();
            System.out.print(" " + var3);
        }

    }
}
