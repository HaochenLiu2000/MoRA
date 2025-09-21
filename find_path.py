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


if __name__ == "__main__":
    subgraph_triplets = [
        ["A", "likes", "B"],
        ["B", "knows", "C"],
        ["C", "helps", "A"],
        ["C", "supports", "D"],
        ["D", "related_to", "E"],
    ]
    
    query_entities = {"A"}
    answer_entities = {"E"}
    max_hops = 3
    
    positive_entities, positive_triplet_indices = find_paths_in_subgraph(
        subgraph_triplets, query_entities, answer_entities, max_hops
    )
    
    print("Positive Entities:", positive_entities)
    print("Positive Triplet Indices:", positive_triplet_indices)
    for idx in positive_triplet_indices:
        print("Triplet:", subgraph_triplets[idx])
    