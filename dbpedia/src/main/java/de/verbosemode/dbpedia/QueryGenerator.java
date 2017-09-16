package de.verbosemode.dbpedia;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import de.verbosemode.dbpedia.util.Entity;
import de.verbosemode.dbpedia.util.Json;
import lombok.Getter;
import lombok.SneakyThrows;
import org.apache.commons.io.FileUtils;
import org.apache.jena.base.Sys;
import org.apache.jena.ext.com.google.common.collect.Lists;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.transport.client.PreBuiltTransportClient;

import java.io.File;
import java.net.InetAddress;
import java.nio.charset.Charset;
import java.text.MessageFormat;
import java.util.*;
import java.util.stream.Collectors;


public class QueryGenerator {

    private Collection<Entity> objects;
    private Collection<Entity> props;

    private String compStr;
    private String properties;

    @SneakyThrows
    public QueryGenerator(Collection<Entity> objects, Collection<Entity> props) {
        this.objects = objects;
        this.props = props;
    }

    public String objects(int limit) {

        List<String> strings = objects.stream().filter(s -> s.getLabel().length() > 0).map(s -> "\"" + s.getLabel() + "\"").distinct().collect(Collectors.toList());

        List<List<String>> partition = Lists.partition(strings, strings.size() / 2);
        List<String> a = Lists.newArrayList(Iterables.limit(partition.get(0), limit));
        List<String> b = Lists.newArrayList(Iterables.limit(partition.get(1), limit));


        List<String> props = this.props.stream().distinct().limit(limit).filter(s -> s.getLabel().length() > 0).map(s -> "\"" + s.getLabel() + "\"").collect(Collectors.toList());


        return "( " + Joiner.on(" OR ").join(a) + ")";
    }

    @SneakyThrows
    public static void main(String... args) {

        TransportClient client = new PreBuiltTransportClient(Settings.EMPTY)
                .addTransportAddress(new InetSocketTransportAddress(InetAddress.getByName("https://9d0fec4462c1b8723270b0099e94777e.europe-west1.gcp.cloud.es.io"), 9243));

        SearchResponse searchResponse = client.prepareSearch("commoncrawl").setQuery(QueryBuilders.termQuery("text", "the")).execute().actionGet();
        //    for (Entity entity : Json.read("arg/concept.json")) {
        //        System.out.println("## " + entity.getLabel());
        String entity = "movie";
        Collection<Entity> read = Json.read("arg/obj/" + entity.replaceAll(" ", "_") + ".json");
        Collection<Entity> props = Json.read("arg/prop/" + entity.replaceAll(" ", "_") + ".json");
        if (!read.isEmpty()) {
            QueryGenerator generator = new QueryGenerator(read, props);
            String objects = generator.objects(100);
            int length = objects.length();
            System.out.println(length);

        }
        //   }


    }
}
