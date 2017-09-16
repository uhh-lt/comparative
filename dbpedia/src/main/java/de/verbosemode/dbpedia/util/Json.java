package de.verbosemode.dbpedia.util;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.SneakyThrows;
import lombok.extern.java.Log;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.Collection;
import java.util.Collections;

@Log
public class Json {

    private static ObjectMapper mapper = new ObjectMapper();

    @SneakyThrows
    public static void write(String path, Collection<Entity> data) {
        log.info("Write " + path);
        FileUtils.write(new File(path), mapper.writeValueAsString(data), false);
    }

    @SneakyThrows
    public static Collection<Entity> read(String path) {
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
}
