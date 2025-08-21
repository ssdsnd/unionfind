import sys
import csv

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each element is its own parent initially
        self.rank = [0] * n  # Rank of each node, initially 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            # Union by rank to keep the tree shallow
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def print_tree(self):
        print(f"Parent array: {self.parent}")
        print(f"Rank array: {self.rank}")

def read_rules_from_file(filename):
    """
    Read rules from a CSV or text file. The rules file format is expected to have
    the following format for each row: A,B,C,|,target
    """
    rules = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 4:
                continue
            inputs, target = row[:-1], row[-1]
            # Handle the case where inputs might be '1,2,3' etc.
            inputs = tuple(input.strip() for input in inputs)
            rules.append((*inputs, '|', target.strip()))
    return rules

def parse_inputs_from_args(args):
    """
    Parse the input values from the command-line arguments.
    """
    if len(args) == 0:
        print("Please provide input values.")
        sys.exit(1)
    
    # Input format: e.g. "1,2,3" -> ['1', '2', '3']
    inputs = tuple(arg.strip() for arg in args)
    return inputs

def parse_rules(rules):
    num_columns = len(rules[0]) - 2  # Remove target and '|'
    input_columns = [chr(65 + i) for i in range(num_columns)]  # ['A', 'B', 'C', ...]
    inputs_map = {col: {} for col in input_columns}
    all_inputs = []

    for rule in rules:
        split_idx = rule.index('|')
        inputs, target = rule[:split_idx], rule[split_idx+1:]

        for idx, input_val in enumerate(inputs):
            if input_val != '*':
                col = input_columns[idx]
                if input_val not in inputs_map[col]:
                    inputs_map[col][input_val] = len(all_inputs)
                    all_inputs.append(input_val)

    uf = UnionFind(len(all_inputs))

    for rule in rules:
        split_idx = rule.index('|')
        inputs, target = rule[:split_idx], rule[split_idx+1:]

        for idx, input_val in enumerate(inputs):
            if input_val != '*':
                input_index = inputs_map[input_columns[idx]].get(input_val, -1)
                if input_index != -1:
                    uf.union(input_index, input_index)
            else:
                for col in input_columns:
                    for val in inputs_map[col]:
                        uf.union(input_index, inputs_map[col][val])

    return uf, all_inputs, inputs_map

def find_best_and_partial_matches(uf, inputs_map, rules, input_values):
    matches = []
    input_columns = list(inputs_map.keys())

    for rule in rules:
        split_idx = rule.index('|')
        inputs, target = rule[:split_idx], rule[split_idx+1:]

        root_set = set()
        
        # For each input, check if it is a wildcard or an exact match
        for idx, input_val in enumerate(inputs):
            if input_val != '*':
                input_index = inputs_map[input_columns[idx]].get(input_val, -1)
                if input_index != -1:
                    root_set.add(uf.find(input_index))
            else:
                root_set.add(None)

        # If all inputs belong to the same root or are wildcards, they match
        if len(root_set) == 1:
            matches.append((inputs, target))
    
    # Now, match the provided input_values to the rules
    matched_rules = []
    for match in matches:
        inputs, target = match
        if all(value == '*' or value == input_values[idx] for idx, value in enumerate(inputs)):
            matched_rules.append((inputs, target))
    
    return matched_rules

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <rules_file> <input1> <input2> ... <inputN>")
        sys.exit(1)

    rules_file = sys.argv[1]
    input_values = parse_inputs_from_args(sys.argv[2:])

    # Read the rules from the provided file
    rules = read_rules_from_file(rules_file)

    # Process the rules and build the Union-Find structure
    uf, all_inputs, inputs_map = parse_rules(rules)
    uf.print_tree()  # Print the union-find tree structure

    # Find matches based on the provided input values
    matches = find_best_and_partial_matches(uf, inputs_map, rules, input_values)
    
    print("Matched Rules (Exact/Partial Matches):")
    for inputs, target in matches:
        print(f"Inputs: {inputs}, Target: {target}")

if __name__ == "__main__":
    main()

