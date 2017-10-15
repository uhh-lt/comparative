package de.verbosemode.dbpedia.util;

import com.google.common.base.MoreObjects;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.apache.jena.rdf.model.Resource;

import java.util.Comparator;
import java.util.Objects;

@Builder
@Data
@AllArgsConstructor
@EqualsAndHashCode(of = "uri")
public class Entity implements Comparable<Entity> {

    private String uri;
    private String label;

    @Override
    public String toString() {
        return label;
    }


    @Override
    public int compareTo(Entity o) {
        return Objects.compare(label, o.label, String.CASE_INSENSITIVE_ORDER);
    }
}
