package de.verbosemode.dbpedia.util;

import lombok.SneakyThrows;
import org.apache.jena.ext.com.google.common.collect.Lists;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.sparql.engine.http.QueryEngineHTTP;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DBPQueries {


    // TODO: use a local dbpedia copy
    private static QueryEngineHTTP buildQueryEngine(Query query) {
        QueryEngineHTTP qexec = (QueryEngineHTTP) QueryExecutionFactory.sparqlService("http://dbpedia.org/sparql", query);
        qexec.addParam("timeout", "10000");
        qexec.setDefaultGraphURIs(Lists.newArrayList("http://dbpedia.org"));
        return qexec;
    }

    public static ResultSet resultSet(Query query) {
        return buildQueryEngine(query).execSelect();
    }

    public static Collection<QuerySolution> list(Query query) {
        ResultSet resultSet = buildQueryEngine(query).execSelect();
        List<QuerySolution> list = new ArrayList<>();
        while (resultSet.hasNext()) {
            list.add(resultSet.next());
        }
        return list;
    }

    public static Stream<QuerySolution> stream(Query query) {
        return list(query).stream();
    }

    public static List<Entity> entities(Query query, String varName){
        return DBPQueries.stream(query).map(s -> s.getResource(varName))
                .map(r -> new Entity(r.getURI(), DBPQueries.getLabel(r))).collect(Collectors.toList());
    }


    public static String getLabel(Resource resource) {
        String label = resource.getURI();
        ResultSet result = resultSet(QueryString.builder().prefix(QueryString.RDFS).query("?val where { <" + resource.getURI() + "> rdfs:label ?val \n" +
                "FILTER(langMatches(lang(?val),\"EN\")) }").build().toQuery());
        if (result.hasNext()) {
            label = result.next().getLiteral("?val").getString();
        }
        return label;
    }


}
