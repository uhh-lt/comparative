package de.verbosemode.dbpedia.util;


import lombok.*;
import lombok.extern.java.Log;
import lombok.extern.log4j.Log4j;
import lombok.extern.slf4j.Slf4j;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;

import java.util.AbstractMap;
import java.util.List;
import java.util.Map;


@Builder
@AllArgsConstructor
@Log
public class QueryString {

    public static Prefix RDF = Prefix.of().prefix("rdf").url("http://www.w3.org/1999/02/22-rdf-syntax-ns#").build();
    public static Prefix OWL = Prefix.of().prefix("owl").url("http://www.w3.org/2002/07/owl#").build();
    public static Prefix DBO = Prefix.of().prefix("dbo").url("http://dbpedia.org/ontology/").build();
    public static Prefix RDFS = Prefix.of().prefix("rdfs").url("http://www.w3.org/2000/01/rdf-schema#").build();
    // namespaces: http://eo.dbpedia.org/sparql?nsdecl
    @Singular
    private List<Prefix> prefixes;
    private boolean distinct;
    private int limit;
    private String query;


    public Query toQuery() {
        StringBuffer buffer = new StringBuffer();
        prefixes.forEach(e -> buffer.append("PREFIX ").append(e.getPrefix()).append(": <").append(e.getUrl()).append(">\n"));
        buffer.append("SELECT ");
        if (distinct) {
            buffer.append("DISTINCT ");
        }
        buffer.append(query);
        if (limit > 0) {
            buffer.append(" LIMIT ").append(limit);
        }

        Query query = QueryFactory.create(buffer.toString());
        System.out.println(buffer.toString());
        return query;

    }

    @Data
    @Builder(builderMethodName = "of")
    public static class Prefix {
        public String prefix;
        public String url;
    }

}
