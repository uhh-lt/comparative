package de.verbosemode.dbpedia;

import de.verbosemode.dbpedia.util.DBPQueryExec;
import de.verbosemode.dbpedia.util.QueryString;
import lombok.extern.slf4j.Slf4j;
import org.apache.jena.query.*;
import org.apache.jena.sparql.engine.http.QueryEngineHTTP;


@Slf4j
public class Scraper {


    public static void main(String[] args) {

        Query topCategories = QueryString.builder().prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
                .prefix("owl", "http://www.w3.org/2002/07/owl#").query("?Concept where {?Concept rdf:type owl:Class}")
                .distinct(true).limit(5).build().toQuery();

        ResultSet resultSet = DBPQueryExec.exec(topCategories).execSelect();
        ResultSetFormatter.out(System.out, resultSet, topCategories);
    }
}
