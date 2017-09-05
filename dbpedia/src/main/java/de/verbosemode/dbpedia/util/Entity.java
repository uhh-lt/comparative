package de.verbosemode.dbpedia.util;

import com.google.common.base.MoreObjects;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.apache.jena.rdf.model.Resource;

@Builder
@Data
@AllArgsConstructor
public class Entity {

    private String uri;
    private String label;

    @Override
    public String toString() {
        return label;
    }
}
