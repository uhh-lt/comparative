package de.verbosemode.dbpedia.util;

import lombok.SneakyThrows;
import org.apache.jena.ext.com.google.common.collect.Lists;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.sparql.engine.http.QueryEngineHTTP;

public class DBPQueryExec {


    // TODO: use a local dbpedia copy
    private static QueryEngineHTTP buildQueryEngine(Query query) {
        QueryEngineHTTP qexec = (QueryEngineHTTP) QueryExecutionFactory.sparqlService("http://dbpedia.org/sparql", query);
        qexec.addParam("timeout", "10000");
        qexec.setDefaultGraphURIs(Lists.newArrayList("http://dbpedia.org"));
        return qexec;
    }

    public static ResultSet select(Query query) {
        return buildQueryEngine(query).execSelect();
    }

    public static String getLabel(Resource resource) {
        return  "";
    }


}
