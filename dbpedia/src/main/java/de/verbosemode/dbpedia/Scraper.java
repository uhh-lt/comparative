package de.verbosemode.dbpedia;

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import de.verbosemode.dbpedia.util.DBPQueries;
import de.verbosemode.dbpedia.util.Entity;
import de.verbosemode.dbpedia.util.QueryString;
import lombok.extern.java.Log;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.*;
import org.apache.jena.vocabulary.OWL;

import java.util.*;
import java.util.stream.Collectors;

@Log
public class Scraper {
    private final static String PROPERTY_NAMESPACE = "http://dbpedia.org/property/";
    private final Model model = ModelFactory.createOntologyModel();

    QueryString.QueryStringBuilder namespace = QueryString.builder().prefix(QueryString.RDF)
            .prefix(QueryString.OWL).prefix(QueryString.DBO).prefix(QueryString.RDFS);

    public final List<Entity> getConcepts(int limit) {
        List<Entity> secondLevel = new ArrayList<>();
        Query flQuery = namespace.query("?subj where {?subj rdfs:subClassOf owl:Thing}")
                .distinct(true).limit(limit).build().toQuery();
        return DBPQueries.entities(flQuery, "?subj");
    }


    public final List<Entity> getObjects(Entity concept, int limit) {
        Query children = namespace.query("?subj WHERE {?subj a <" + concept.getUri() + ">}").limit(limit).build().toQuery();
        return DBPQueries.entities(children, "?subj");
    }

    public final List<Entity> getProperties(Entity entity, int limit) {

        String org =" ?prop (count(?prop) as ?NPROP)  where { ?subj a <"+entity.getUri()+">\n" +
                "   { \n" +
                "select * where { ?subj ?prop ?val\n" +
                "FILTER(STRSTARTS(STR(?prop), \"http://dbpedia.org/property\"))\n" +
                " } \n" +
                "} \n" +
                " } GROUP BY ?prop ORDER BY DESC (?NPROP)";

        Query query = namespace.query(org
              ).limit(limit).build().toQuery();


        return DBPQueries.entities(query, "?prop");


    }

    public static void main(String... args) {
        Scraper scraper = new Scraper();
        List<Entity> concepts = scraper.getConcepts(3);
        Map<String, List<Entity>> objects = new HashMap<>();
        Map<String, List<Entity>> properties = new HashMap<>();
        for (Entity concept : concepts) {
            objects.put(concept.getLabel(), scraper.getObjects(concept, 5));
            properties.put(concept.getLabel(), scraper.getProperties(concept, 10));
        }

        Joiner.MapJoiner mapJoiner = Joiner.on("\n").withKeyValueSeparator("=");
        System.out.println(mapJoiner.join(properties));

    }


}
