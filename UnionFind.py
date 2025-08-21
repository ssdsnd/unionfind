class UnionFind:
    def __init__(self, n):
        # Initialize parent and rank arrays
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        # Path compression to make future queries faster
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # Union by rank to keep the tree flat
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def parse_rules(rules):
    # Automatically determine the number of input columns from the first rule
    num_columns = len(rules[0]) - 2  # Subtracting 2 to remove target and '|'

    # Dynamically create input columns based on the number of input columns
    input_columns = [chr(65 + i) for i in range(num_columns)]  # Generates ['A', 'B', 'C', ...]
    
    # Create the input mapping and all unique inputs
    inputs_map = {col: {} for col in input_columns}
    all_inputs = []

    # Create a map of inputs and targets
    for rule in rules:
        inputs, target = rule[:-1], rule[-1]
        for idx, input_val in enumerate(inputs):
            if input_val != '*':  # Exclude wildcards initially
                col = input_columns[idx]
                if input_val not in inputs_map[col]:
                    inputs_map[col][input_val] = len(all_inputs)
                    all_inputs.append(input_val)

    # Create UnionFind structure for all unique inputs
    uf = UnionFind(len(all_inputs))

    # Now process the rules to form unions based on exact matches or wildcard matches
    for rule in rules:
        inputs, target = rule[:-1], rule[-1]
        for idx, input_val in enumerate(inputs):
            if input_val != '*':  # Only handle non-wildcard inputs
                input_index = inputs_map[input_columns[idx]][input_val]
                # For wildcard `*`, union it with all other inputs in that column
                if input_val == '*':
                    for other_val in inputs_map[input_columns[idx]]:
                        uf.union(input_index, inputs_map[input_columns[idx]][other_val])

    return uf, all_inputs, inputs_map

def find_best_and_partial_matches(uf, inputs_map, rules):
    matches = []
    for rule in rules:
        inputs, target = rule[:-1], rule[-1]
        root_set = set(uf.find(inputs_map[input_columns[idx]].get(input_val, -1)) 
                       for idx, input_val in enumerate(inputs) if input_val != '*')

        # If all inputs belong to the same root, they are in the same group
        if len(root_set) == 1:
            matches.append((inputs, target))
    return matches

# Sample rules and targets
rules = [
    ('1', '2', '3', '|', '10'),
    ('1', '2', '*', '|', '20'),
    ('1', '*', '*', '|', '30'),
    ('*', '2', '5', '|', '50'),
]

# Main Program
def main():
    uf, all_inputs, inputs_map = parse_rules(rules)
    
    # Test the matching process
    matches = find_best_and_partial_matches(uf, inputs_map, rules)
    print("Matched Rules (Exact/Partial Matches):")
    for inputs, target in matches:
        print(f"Inputs: {inputs}, Target: {target}")

if __name__ == "__main__":
    main()

