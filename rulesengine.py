import csv
import numpy as np
from collections import defaultdict
import time

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)  # Maps value (str) to next node
        self.rules = []  # List of (output, specificity, rule_idx, conditions)

class RulesEngine:
    def __init__(self, csv_file):
        self.rules = None
        self.trie = TrieNode()
        self.num_columns = 0
        self.load_rules(csv_file)

    def load_rules(self, csv_file):
        start_time = time.time()
        # Read CSV into NumPy array
        rules_list = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header
            self.num_columns = len(header) - 1  # Exclude Output column
            for row in reader:
                conditions = row[:-1]  # All columns except Output
                output = row[-1]  # Output column
                # Split list-based conditions (e.g., VG:EUR:PUBLIC -> {VG, EUR, PUBLIC})
                typed_conditions = []
                for c in conditions:
                    if ':' in c and c != '*':
                        typed_conditions.append(set(c.split(':')))
                    else:
                        typed_conditions.append(str(c))  # Convert to string for consistency
                rules_list.append(typed_conditions + [output])
        
        self.rules = np.array(rules_list, dtype=object)
        print(f"Loaded {len(self.rules)} rules with {self.num_columns} columns in {time.time() - start_time:.2f}s")
        
        # Build trie
        start_time = time.time()
        for idx, rule in enumerate(self.rules):
            conditions = rule[:-1]
            output = rule[-1]
            specificity = sum(1 for c in conditions if c != '*')
            # Create all possible paths for this rule
            def build_path(node, depth, path_conditions):
                if depth == self.num_columns:
                    node.rules.append((output, specificity, idx, conditions))
                    print(f"Debug: Added rule {idx} at depth {depth}: {path_conditions}, Output={output}, Specificity={specificity}")
                    return
                c = conditions[depth]
                values = c if isinstance(c, set) else {str(c)}
                for v in values:
                    new_path = path_conditions + [v]
                    build_path(node.children[v], depth + 1, new_path)
            
            build_path(self.trie, 0, [])
        print(f"Built trie in {time.time() - start_time:.2f}s")

        # Pretty print the trie
        print("\nPretty Printing Trie:")
        self.pretty_print_trie(self.trie)

    def evaluate(self, input_vector):
        if len(input_vector) != self.num_columns:
            raise ValueError(f"Input must have {self.num_columns} values")
        
        # Convert input to strings for consistency
        typed_input = [str(v) for v in input_vector]
        print(f"Debug: Evaluating input: {typed_input}")
        
        start_time = time.time()
        # Collect all matching rules
        applicable_rules = []
        def traverse(node, depth):
            if depth == self.num_columns:
                if node.rules:
                    print(f"Debug: Found rules at depth {depth}: {[r[2] for r in node.rules]}")
                applicable_rules.extend(node.rules)
                return
            value = typed_input[depth]
            print(f"Debug: At depth {depth}, checking value {value}")
            # Follow exact match or wildcard
            if value in node.children:
                print(f"Debug: Following path for value {value}")
                traverse(node.children[value], depth + 1)
            if '*' in node.children:
                print(f"Debug: Following wildcard path")
                traverse(node.children['*'], depth + 1)
        
        traverse(self.trie, 0)
        
        if not applicable_rules:
            print("Debug: No matching rules found")
            return None, []
        
        # Best match: Rule with highest specificity
        best_match = max(applicable_rules, key=lambda x: x[1])
        # Partial matches: All matching rules with details
        partial_matches = [
            {
                "output": output,
                "specificity": specificity,
                "rule_index": idx,
                "conditions": [list(c) if isinstance(c, set) else c for c in conditions.tolist()]
            }
            for output, specificity, idx, conditions in applicable_rules
        ]
        
        print(f"Query time: {(time.time() - start_time)*1000:.2f}ms")
        return best_match[0], partial_matches

    def pretty_print_trie(self, node, depth=0, prefix=""):
        """Pretty print the Trie structure."""
        indent = "  " * depth
        for value, child in node.children.items():
            rule_info = [f"Rule {r[2]}: Output={r[0]}, Specificity={r[1]}" for r in child.rules]
            print(f"{indent}{prefix}Value: {value}, Rules: {rule_info if rule_info else 'None'}")
            self.pretty_print_trie(child, depth + 1, f"{prefix}{value} -> ")

# Example usage
if __name__ == "__main__":
    # Simulate CSV with sample rules
    csv_content = """Client,REGION,BU,Trader_Domicile,Output
*,NA,EQ,HK,rule1
*,NA,EQ,JP,rule2
1,NA,EQ,*,rule3"""

    with open("rules.csv", "w") as f:
        f.write(csv_content)

    # Initialize rules engine
    engine = RulesEngine("rules.csv")

    # Test case 1: Input matches rule 1 and 2 (7 columns)
    test_input = ["1", "NA", "EQ", "HK"]  # 7 values, strings
    best_output, partial_matches = engine.evaluate(test_input)
    print(f"\nInput: {test_input[:4]}... (7 columns)")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions'][:5]}...")




'''
    # Simulate CSV with sample rules
    csv_content = """Client,LE,Flag1,Trader_Domicile,Flag2,Output,Group1,Group2
*,NA,1,*,*,"A|B|C",Stocks,Equities
001,NA,1,*,*,"A:soft|B|C",Stocks,Equities
*,VG:EUR:PUBLIC,*,EU,*,"C|D",Stocks,Equities
*,VG:EUR:PUBLIC,*,GB,*,"C:hard|D",Stocks,Equities"""
    with open("rules.csv", "w") as f:
        f.write(csv_content)

    # Initialize rules engine
    engine = RulesEngine("rules.csv")

    # Test case 1: Input matches rule 1 and 2 (7 columns)
    test_input = ["001", "NA", "1", "EU", "0", "Stocks", "Equities"]  # 7 values, strings
    best_output, partial_matches = engine.evaluate(test_input)
    print(f"\nInput: {test_input[:5]}... (7 columns)")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions'][:5]}...")

    # Test case 2: Input matches rule 3 and 4
    test_input = ["002", "EUR", "0", "EU", "0", "Stocks", "Equities"]  # 7 values, strings
    best_output, partial_matches = engine.evaluate(test_input)
    print(f"\nInput: {test_input[:5]}... (7 columns)")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions'][:5]}...")

'''