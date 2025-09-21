import re
from collections import defaultdict

def find_paths_in_subgraph(subgraph_triplets, query_entities, answer_entities, max_hops):
    
    adjacency = {}
    for idx, (h, _, t) in enumerate(subgraph_triplets):
        if h not in adjacency:
            adjacency[h] = []
        if t not in adjacency:
            adjacency[t] = []
        adjacency[h].append((t, idx))
        adjacency[t].append((h, idx))

    positive_entities = set()
    positive_triplet_indices = set()

    def dfs(current_entity, visited_entities, visited_triplets, depth):
        if depth > max_hops:
            return False
        if current_entity in answer_entities:
            positive_entities.update(visited_entities)
            positive_triplet_indices.update(visited_triplets)
            return True

        found = False
        for neighbor_tuple in adjacency.get(current_entity, []):
            neighbor, triplet_idx = neighbor_tuple
            if triplet_idx not in visited_triplets:
                if dfs(
                    neighbor,
                    visited_entities | {neighbor},
                    visited_triplets | {triplet_idx},
                    depth + 1
                ):
                    found = True
        return found

    for query_entity in query_entities:
        dfs(query_entity, {query_entity}, set(), 0)

    return positive_entities, positive_triplet_indices


def is_unknown_entity(ent):
    return re.match(r'^<unknown\d+>$', ent) or re.match(r'^<node_.*>$', ent)



def merge_structural_unknowns(
    triplets, query_entities, unknown_counter_merge, positive_triplet_indices=None, max_hops=3, min_merge_count=2
):
    protected_entities = set()
    if positive_triplet_indices is not None:
        for idx in positive_triplet_indices:
            h, _, t = triplets[idx]
            protected_entities.add(h)
            protected_entities.add(t)

    neighbors = defaultdict(list)
    for idx, (h, r, t) in enumerate(triplets):
        neighbors[h].append((t, r, "forward"))
        neighbors[t].append((h, r, "backward"))

    visited = set(query_entities)
    current_layer = set(query_entities)
    merged_map = {}

    for hop in range(1, max_hops + 1):
        next_layer = set()
        next_layer_all = set()
        merge_groups = defaultdict(list)
        
        note=unknown_counter_merge[0]
        for node in current_layer:
            for neighbor, rel, direction in neighbors[node]:
                if neighbor is None:
                    continue
                if is_unknown_entity(neighbor) and neighbor not in protected_entities and neighbor not in visited:
                    merge_groups[(node, rel, direction)].append(neighbor)
                    if len(merge_groups[(node, rel, direction)])==2:
                        next_layer.add(f"<merged_{note}>")
                        note+=1
                elif neighbor not in visited:
                    next_layer.add(neighbor)
                if neighbor not in visited:
                    next_layer_all.add(neighbor)

        for (src, rel, direction), candidates in merge_groups.items():
            unique = list(set(candidates))
            if len(unique) >= min_merge_count:
                merged_node = f"<merged_{unknown_counter_merge[0]}>"
                unknown_counter_merge[0]+=1
                for unk in unique:
                    merged_map[unk] = merged_node

                for unk in unique:
                    for neighbor, r, dir in neighbors.get(unk, []):
                        neighbors[merged_node].append((neighbor, r, dir))
                        neighbors[neighbor].append((merged_node, r, "backward" if dir == "forward" else "forward"))
        visited.update(current_layer)
        visited.update(next_layer)
        visited.update(next_layer_all)
        current_layer = next_layer
        

    new_triplets = set()
    for h, r, t in triplets:
        h_new = merged_map.get(h, h)
        t_new = merged_map.get(t, t)
        new_triplets.add((h_new, r, t_new))

    return list(new_triplets), merged_map

def find_pruned_paths_in_subgraph(subgraph_triplets, query_entities, answer_entities, unknown_counter_merge, max_hops):
    positive_entities, positive_triplet_indices = find_paths_in_subgraph(
        subgraph_triplets, query_entities, answer_entities, max_hops
    )
    subgraph_triplets, merged_map = merge_structural_unknowns(
        subgraph_triplets, query_entities, unknown_counter_merge, positive_triplet_indices=positive_triplet_indices, max_hops=max_hops, min_merge_count=2
    )
    subgraph_triplets = [list(t) for t in subgraph_triplets]
    positive_entities, positive_triplet_indices = find_paths_in_subgraph(
        subgraph_triplets, query_entities, answer_entities, max_hops
    )
    
    
    return subgraph_triplets, positive_entities, positive_triplet_indices

if __name__ == "__main__":
    subgraph_triplets = [
    ("A", "r1", "<unknown1>"),
    ("A", "r1", "<unknown2>"),
    ("A", "r1", "<unknown30>"),
    ("A", "r1", "<unknown40>"),
    ("<unknown300>", "r100", "<unknown30>"),
    ("<unknown400>", "r100", "<unknown40>"),
    ("<unknown3>", "r1", "A"),
    ("<unknown4>", "r1", "A"),
    ("B", "r1", "<unknown1>"),
    ("B", "r1", "<unknown2>"),
    ("<unknown3>", "r1", "B"),
    ("<unknown4>", "r1", "B"),
    ("<unknown1>", "r2", "<unknown5>"),
    ("<unknown2>", "r2", "<unknown6>"),
    ("<unknown100>", "r2", "<unknown2>"),
    ("<unknown200>", "r2", "<unknown2>"),
    ("<unknown3>", "r2", "<unknown7>"),
    ("<unknown4>", "r2", "<unknown8>"),
    ("<unknown5>", "r3", "E"),
    ("<unknown6>", "r3", "E"),
    ("<unknown7>", "r3", "E"),
    ("<unknown8>", "r3", "E"),
    ]

    query_entities = {"A"}
    answer_entities = {"E"}
    max_hops = 3

    positive_entities, positive_triplet_indices = find_paths_in_subgraph(
        subgraph_triplets, query_entities, answer_entities, max_hops
    )
    unknown_counter_merge=[1]
    merged_triplets, merged_map = merge_structural_unknowns(
        subgraph_triplets, query_entities, unknown_counter_merge, positive_triplet_indices=positive_triplet_indices, max_hops=max_hops, min_merge_count=1
    )
    
    positive_entities_new, positive_triplet_indices_new = find_paths_in_subgraph(
        merged_triplets, query_entities, answer_entities, max_hops
    )
    