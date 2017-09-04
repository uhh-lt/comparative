package de.verbosemode.dbpedia.util;

import lombok.SneakyThrows;
import org.apache.jena.ext.com.google.common.collect.Lists;
import org.apache.jena.query.*;
import org.apache.jena.sparql.engine.http.QueryEngineHTTP;

public class DBPQueryExec {

    @SneakyThrows
    public static QueryEngineHTTP exec(QueryString queryString) {
        return exec(queryString.toQuery());
    }

    @SneakyThrows
    public static QueryEngineHTTP exec(Query query) {
        QueryEngineHTTP qexec = (QueryEngineHTTP) QueryExecutionFactory.sparqlService("http://dbpedia.org/sparql", query);
        qexec.addParam("timeout", "10000");
        qexec.setDefaultGraphURIs(Lists.newArrayList("http://dbpedia.org"));
        return qexec;
    }


}
