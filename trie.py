import csv
import numpy as np
from collections import defaultdict
import time

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)  # Maps value (int or '*') to next node
        self.rules = []  # List of (output, specificity, rule_idx, rule_conditions)

class RulesEngine:
    def __init__(self, csv_file):
        self.rules = None
        self.trie = TrieNode()
        self.load_rules(csv_file)

    def load_rules(self, csv_file):
        start_time = time.time()
        # Read CSV into NumPy array
        rules_list = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                conditions = row[:-1]  # C1,C2,...,C40
                output = int(row[-1])  # Output
                rules_list.append(conditions + [output])
        
        self.rules = np.array(rules_list, dtype=object)
        print(f"Loaded {len(self.rules)} rules in {time.time() - start_time:.2f}s")
        
        # Build trie
        start_time = time.time()
        for idx, rule in enumerate(self.rules):
            conditions = rule[:-1]
            output = rule[-1]
            specificity = sum(1 for c in conditions if c != '*')
            node = self.trie
            for c in conditions:
                node = node.children[c]
            node.rules.append((output, specificity, idx, conditions))
        print(f"Built trie in {time.time() - start_time:.2f}s")

    def evaluate(self, input_vector):
        if len(input_vector) != 40:
            raise ValueError("Input must have 40 values")
        
        start_time = time.time()
        # Collect all matching rules
        applicable_rules = []
        def traverse(node, depth):
            if depth == 40:
                applicable_rules.extend(node.rules)
                return
            value = str(input_vector[depth])
            # Follow exact match or wildcard
            if value in node.children:
                traverse(node.children[value], depth + 1)
            if '*' in node.children:
                traverse(node.children['*'], depth + 1)
        
        traverse(self.trie, 0)
        
        if not applicable_rules:
            return None, []
        
        # Best match: Rule with highest specificity
        best_match = max(applicable_rules, key=lambda x: x[1])
        # Partial matches: All matching rules with details
        partial_matches = [
            {
                "output": output,
                "specificity": specificity,
                "rule_index": idx,
                "conditions": conditions.tolist()
            }
            for output, specificity, idx, conditions in applicable_rules
        ]
        
        print(f"Query time: {(time.time() - start_time)*1000:.2f}ms")
        return best_match[0], partial_matches

# Example usage
if __name__ == "__main__":
    # Simulate CSV (replace with actual 1M-row CSV)
    csv_content = """C1,C2,C3,C4,...,C40,Output
1,2,3,*,...,*,10
1,2,4,*,...,*,20
1,2,*,*,...,*,30
*,3,*,*,...,*,50"""
    csv_content = csv_content.replace("C4,...,C40", ",".join([f"C{i}" for i in range(4, 41)]))
    with open("rules.csv", "w") as f:
        f.write(csv_content)

    # Initialize rules engine
    engine = RulesEngine("rules.csv")

    # Test case 1: Input matches rule 1 and 3
    test_input = [1, 2, 3] + ['0'] * 37
    best_output, partial_matches = engine.evaluate(test_input)
    print(f"\nInput: {test_input[:5]}...")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions'][:5]}...")

    # Test case 2: Input matches rule 4
    test_input = [2, 3, 1] + ['0'] * 37
    best_output, partial_matches = engine.evaluate(test_input)
    print(f"\nInput: {test_input[:5]}...")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions'][:5]}...")
