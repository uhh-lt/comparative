package de.verbosemode.dbpedia;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.verbosemode.dbpedia.util.DBPQueries;
import de.verbosemode.dbpedia.util.Entity;
import de.verbosemode.dbpedia.util.QueryString;
import lombok.*;

import lombok.extern.java.Log;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

import org.apache.jena.query.*;

import java.io.File;
import java.util.*;

import java.util.stream.Collectors;

@Log
public class Scraper {
    public static final int MAX_LIMIT = 40000;
    private ObjectMapper mapper = new ObjectMapper();


    private QueryString.QueryStringBuilder namespace = QueryString.builder().prefix(QueryString.RDF)
            .prefix(QueryString.OWL).prefix(QueryString.DBO).prefix(QueryString.RDFS);


    private static final String TOP_LEVEL_THING = "?subj WHERE {?subj rdfs:subClassOf owl:Thing}";

    private static final String TOP_LEVEL_CLASS = "?subj WHERE {?subj a owl:Class}";


    @SneakyThrows
    public void write(String path, Collection<Entity> data) {
        log.info("Write " + path);
        FileUtils.write(new File(path), mapper.writeValueAsString(data), false);
    }

    @SneakyThrows
    public Collection<Entity> read(String path) {
        File file = new File(path);
        log.info("Read " + path);
        if (file.exists()) {
            String saved = FileUtils.readFileToString(file, "utf-8");
            Collection<Entity> entities = mapper.readValue(saved, new TypeReference<Collection<Entity>>() {
            });
            return entities;
        }
        return Collections.emptyList();
    }

    /**
     * gets the first two levels of DBPedia ontology as classes
     *
     * @param limit
     * @return
     */
    @SneakyThrows
    public final Collection<Entity> getConcepts(int limit) {
        Collection<Entity> read = read("arg/concept.json");
        if (!read.isEmpty()) {
            return read;
        } else {
            try {
                Set<Entity> concepts = new TreeSet<>();
                QueryString.QueryStringBuilder base = namespace.distinct(true).limit(limit);


                System.out.println("+- Class\n|");
                Collection<Entity> classes = DBPQueries.entities(base.query(TOP_LEVEL_CLASS).build().toQuery(), "?subj");

                for (Entity entity : classes) {
                    Thread.sleep(1000);
                    System.out.println("|-- " + entity.getLabel());
                    Query slClass = base.query("?subj WHERE {?subj rdfs:subClassOf <" + entity.getUri() + ">}")
                            .build().toQuery();
                    Collection<Entity> childs = DBPQueries.entities(slClass, "?subj");
                    childs.forEach(a -> {
                        System.out.println("|--- " + a.getLabel());
                    });
                    concepts.addAll(childs);
                    if (childs.isEmpty()) {
                        concepts.add(entity);
                    }
                }


                Collection<Entity> things = DBPQueries.entities(base.query(TOP_LEVEL_THING)
                        .build().toQuery(), "?subj");
                System.out.println("|\n|\n+- Thing\n|");
                for (Entity entity : things) {
                    Thread.sleep(1000);
                    System.out.println("|-- " + entity.getLabel());
                    Query slTHing = base.query("?subj WHERE {?subj rdfs:subClassOf <" + entity.getUri() + ">}")
                            .build().toQuery();
                    Collection<Entity> childs = DBPQueries.entities(slTHing, "?subj");
                    childs.forEach(a -> System.out.println("|--- " + a.getLabel()));
                    concepts.addAll(childs);
                    if (childs.isEmpty()) {
                        concepts.add(entity);
                    }
                }
                write("arg/concept.json", concepts);
                return concepts;

            } catch (Exception e) {
                log.severe(e.getLocalizedMessage());
                return Collections.emptyList();
            }
        }
    }


    public final Collection<Entity> getObjects(Entity concept, int limit) {
        Query children = namespace.query("?subj WHERE {?subj a <" + concept.getUri() + ">}").limit(limit).build().toQuery();
        return DBPQueries.entities(children, "?subj");
    }

    public final Collection<Entity> getProperties(Entity entity, int limit) {
        String org = "?prop (count(?prop) as ?NPROP)  where { ?subj a <" + entity.getUri() + "> { \n" +
                "select * where { ?subj ?prop ?val\n" +
                "FILTER(STRSTARTS(STR(?prop), \"http://dbpedia.org/property\"))\n" +
                " } \n" +
                "} \n" +
                " } GROUP BY ?prop ORDER BY DESC (?NPROP)";

        Query query = namespace.query(org).limit(limit).build().toQuery();
        return DBPQueries.entities(query, "?prop").stream().filter(o -> !o.getLabel().startsWith("image")).collect(Collectors.toSet());


    }

    /**
     * Removes short strings and double spaces
     *
     * @param list
     * @return
     */
    private Collection<Entity> clear(Collection<Entity> list) {

        Iterator<Entity> it = list.iterator();
        List<Entity> cleaned = new ArrayList<>();
        while (it.hasNext()) {
            Entity entity = it.next();
            String newLAbel = entity.getLabel().replaceAll("\\(.*\\)", "")
                    .replaceAll("\\s{2,}", " ").trim();

            entity.setLabel(newLAbel);

            if (newLAbel.length() > 3) {

                cleaned.add(entity);
            }
        }
        return cleaned;
    }


    @SneakyThrows
    public static void main(String... args) {
        Scraper scraper = new Scraper();

        Collection<Entity> concepts = scraper.getConcepts(Scraper.MAX_LIMIT);

        Map<String, Collection<Entity>> objects = new HashMap<>();
        Map<String, Collection<Entity>> properties = new HashMap<>();

        for (Entity concept : concepts) {
            System.out.println(" Process  " + concept.getLabel());
            try {
                long start = System.nanoTime();
                if (scraper.read("arg/obj/" + concept.getLabel().replaceAll(" ", "_") + ".json").isEmpty()) {


                    Collection<Entity> obj = scraper.getObjects(concept, Scraper.MAX_LIMIT);
                    objects.put(concept.getUri(), obj);
                    Thread.sleep(1000);
                    Collection<Entity> prop = scraper.getProperties(concept, Scraper.MAX_LIMIT);
                    properties.put(concept.getUri(), prop);
                    Thread.sleep(3000);


                    if (obj.size() > 0 && prop.size() > 0) {
                        scraper.write("arg/obj/" + concept.getLabel().replaceAll(" ", "_") + ".json", scraper.clear(obj));
                        scraper.write("arg/prop/" + concept.getLabel().replaceAll(" ", "_") + ".json", scraper.clear(prop));

                    } else {
                        System.out.println(" MISS " + concept.getLabel());
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
            }

        }


    }
}
