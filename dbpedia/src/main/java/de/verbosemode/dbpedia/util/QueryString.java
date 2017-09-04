package de.verbosemode.dbpedia.util;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;
import lombok.extern.java.Log;
import lombok.extern.log4j.Log4j;
import lombok.extern.slf4j.Slf4j;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;

import java.util.Map;

@Log
@Builder
@AllArgsConstructor
public class QueryString {

    // namespaces: http://eo.dbpedia.org/sparql?nsdecl
    @Singular
    private Map<String, String> prefixes;
    private boolean distinct;
    private int limit;
    private String query;


    public Query toQuery() {
        StringBuffer buffer = new StringBuffer();
        prefixes.entrySet().forEach(e -> buffer.append("PREFIX ").append(e.getKey()).append(": <").append(e.getValue()).append(">\n"));
        buffer.append("SELECT ");
        if (distinct) {
            buffer.append("DISTINCT ");
        }
        buffer.append(query);
        if (limit > 0) {
            buffer.append(" LIMIT ").append(limit);
        }

        return QueryFactory.create(buffer.toString());

    }

}
