package de.verbosemode.dbpedia;

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import de.verbosemode.dbpedia.util.DBPQueries;
import de.verbosemode.dbpedia.util.Entity;
import de.verbosemode.dbpedia.util.QueryString;
import lombok.SneakyThrows;
import lombok.extern.java.Log;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.http.util.TextUtils;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.*;
import org.apache.jena.vocabulary.OWL;

import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.stream.Collectors;

@Log
public class Scraper {
    private final static String PROPERTY_NAMESPACE = "http://dbpedia.org/property/";
    private final Model model = ModelFactory.createOntologyModel();

    QueryString.QueryStringBuilder namespace = QueryString.builder().prefix(QueryString.RDF)
            .prefix(QueryString.OWL).prefix(QueryString.DBO).prefix(QueryString.RDFS);

    public final Collection<Entity> getConcepts(int limit) {

        Query flQuery = namespace.query("?subj where {?subj a owl:Class}")
                .distinct(true).limit(limit).build().toQuery();
        Collection<Entity> entities = DBPQueries.entities(flQuery, "?subj");

        return entities;
    }


    public final Collection<Entity> getObjects(Entity concept, int limit) {
        Query children = namespace.query("?subj WHERE {?subj a <" + concept.getUri() + ">}").limit(limit).build().toQuery();
        return DBPQueries.entities(children, "?subj");
    }

    public final Collection<Entity> getProperties(Entity entity, int limit) {

        String org = " ?prop (count(?prop) as ?NPROP)  where { ?subj a <" + entity.getUri() + ">\n" +
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
        Collection<Entity> concepts = scraper.getConcepts(100);

        Map<String, Collection<Entity>> objects = new HashMap<>();
        Map<String, Collection<Entity>> properties = new HashMap<>();

        for (Entity concept : concepts) {
           Collection<Entity> obj = scraper.getObjects(concept, 100);
            objects.put(concept.getUri(), obj);
            Collection<Entity> prop = scraper.getProperties(concept, 20);
            properties.put(concept.getUri(), prop);

            scraper.writeToFile(obj.stream().map(Entity::getLabel).collect(Collectors.toList()), "arg/obj/" + concept.getLabel() + ".txt");
            scraper.writeToFile(prop.stream().map(Entity::getLabel).collect(Collectors.toList()), "arg/prop/" + concept.getLabel() + ".txt");


        }


        Joiner.MapJoiner mapJoiner = Joiner.on("\n").withKeyValueSeparator("=");

        Iterator<Map.Entry<String, Collection<Entity>>> iterator = properties.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, Collection<Entity>> entry = iterator.next();
            if (entry.getValue().isEmpty()) {
                iterator.remove();
                log.info("remove " + entry.getKey());
            }
        }

        System.out.println(mapJoiner.join(properties));

    }

    private List<String> clear(List<String> str) {
        str.removeIf(s -> s.length() < 3 || StringUtils.isNumeric(s));
        str.sort(String.CASE_INSENSITIVE_ORDER);
        return str;
    }

    @SneakyThrows
    public void writeToFile(List<String> list, String path) {
        if (!list.isEmpty()) {
            FileUtils.writeLines(new File(path), clear(list));
        }
    }


}
