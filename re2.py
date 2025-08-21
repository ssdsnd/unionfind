import csv
import numpy as np
import time
from collections import defaultdict

class TrieNode:
    def __init__(self):
        # Use a sorted list of (value, child_node) pairs instead of defaultdict
        self.children = []  # List of (value, TrieNode) tuples
        self.rules = []  # List of (output, specificity, rule_idx, conditions)

class RulesEngine:
    def __init__(self, csv_file):
        self.rules = None
        self.trie = TrieNode()
        self.num_columns = 0
        self.condition_map = {}  # Maps normalized conditions to (output, rule_idx)
        self.load_rules(csv_file)

    def normalize_conditions(self, conditions):
        """Normalize conditions for uniqueness check, converting sets to sorted tuples."""
        normalized = []
        for c in conditions:
            if isinstance(c, set):
                normalized.append(tuple(sorted(c)))  # Sort set for consistency
            else:
                normalized.append(c)
        return tuple(normalized)

    def load_rules(self, csv_file):
        start_time = time.time()
        # Read CSV into NumPy array
        rules_list = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.num_columns = len(header) - 1  # Exclude Output column
            for row_idx, row in enumerate(reader):
                conditions = row[:-1]
                output = row[-1]
                typed_conditions = []
                for c in conditions:
                    if ':' in c and c != '*':
                        typed_conditions.append(set(c.split(':')))
                    else:
                        typed_conditions.append(str(c))
                # Check for duplicates or conflicts
                normalized_conditions = self.normalize_conditions(typed_conditions)
                if normalized_conditions in self.condition_map:
                    prev_output, prev_idx = self.condition_map[normalized_conditions]
                    if prev_output == output:
                        raise ValueError(
                            f"Duplicate rule at row {row_idx + 2}: "
                            f"Conditions={typed_conditions}, Output={output}. "
                            f"Matches rule {prev_idx + 2}."
                        )
                    else:
                        raise ValueError(
                            f"Conflicting rule at row {row_idx + 2}: "
                            f"Conditions={typed_conditions}, Output={output}. "
                            f"Matches rule {prev_idx + 2} with output={prev_output}."
                        )
                self.condition_map[normalized_conditions] = (output, row_idx)
                rules_list.append(typed_conditions + [output])
        
        self.rules = np.array(rules_list, dtype=object)
        print(f"Loaded {len(self.rules)} rules with {self.num_columns} columns in {time.time() - start_time:.2f}s")
        
        # Build array-based trie
        start_time = time.time()
        for idx, rule in enumerate(self.rules):
            conditions = rule[:-1]
            output = rule[-1]
            specificity = sum(1 for c in conditions if c != '*')
            def build_path(node, depth, path_conditions):
                if depth == self.num_columns:
                    node.rules.append((output, specificity, idx, conditions))
                    print(f"Debug: Added rule {idx} at depth {depth}: {path_conditions}, Output={output}, Specificity={specificity}")
                    return
                c = conditions[depth]
                values = c if isinstance(c, set) else {str(c)}
                for v in sorted(values):  # Sort for cache-friendly access
                    # Find or create child node
                    child = next((n for v_, n in node.children if v_ == v), None)
                    if not child:
                        child = TrieNode()
                        node.children.append((v, child))
                        node.children.sort(key=lambda x: x[0])  # Keep children sorted
                    new_path = path_conditions + [v]
                    build_path(child, depth + 1, new_path)
            
            build_path(self.trie, 0, [])
        print(f"Built trie in {time.time() - start_time:.2f}s")
        
        # Pretty print the trie
        print("\nPretty Printing Trie:")
        self.pretty_print_trie(self.trie)

    def pretty_print_trie(self, node, depth=0, prefix=""):
        """Pretty print the array-based Trie structure."""
        indent = "  " * depth
        for value, child in node.children:
            rule_info = [f"Rule {r[2]}: Output={r[0]}, Specificity={r[1]}" for r in child.rules]
            print(f"{indent}{prefix}Value: {value}, Rules: {rule_info if rule_info else 'None'}")
            self.pretty_print_trie(child, depth + 1, f"{prefix}{value} -> ")

    def evaluate(self, input_vector):
        if len(input_vector) != self.num_columns:
            raise ValueError(f"Input must have {self.num_columns} values")
        
        typed_input = [str(v) for v in input_vector]
        print(f"\nDebug: Evaluating input: {typed_input}")
        
        start_time = time.time()
        applicable_rules = []
        def traverse(node, depth):
            if depth == self.num_columns:
                if node.rules:
                    print(f"Debug: Found rules at depth {depth}: {[r[2] for r in node.rules]}")
                applicable_rules.extend(node.rules)
                return
            value = typed_input[depth]
            print(f"Debug: At depth {depth}, checking value {value}, available children: {[v for v, _ in node.children]}")
            # Binary search for exact match (cache-friendly)
            left, right = 0, len(node.children)
            while left < right:
                mid = (left + right) // 2
                child_value, child = node.children[mid]
                if child_value == value:
                    print(f"Debug: Following path for value {value}")
                    traverse(child, depth + 1)
                    break
                elif child_value < value:
                    left = mid + 1
                else:
                    right = mid
            # Check wildcard
            for child_value, child in node.children:
                if child_value == '*':
                    print(f"Debug: Following wildcard path")
                    traverse(child, depth + 1)
                    break
        
        traverse(self.trie, 0)
        
        if not applicable_rules:
            print("Debug: No matching rules found")
            return None, []
        
        best_match = max(applicable_rules, key=lambda x: x[1])
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

# Example usage
if __name__ == "__main__":
    # Simulate CSV with sample rules
    csv_content = """Client,LE,Flag1,Trader_Domicile,Flag2,Group1,Group2,Output
*,NA,1,*,*,Stocks,Equities,rule1
*,NA,*,EU,*,Stocks,Equities,rule2
001,NA,2,*,*,Stocks,Equities,C|D"""
    with open("rules.csv", "w") as f:
        f.write(csv_content)

    # Initialize rules engine with valid CSV
    engine = RulesEngine("rules.csv")

    # Test case 1
    test_input = ["001", "NA", "1", "EU", "0", "Stocks", "Equities"]
    best_output, partial_matches = engine.evaluate(test_input)
    print(f"\nInput: {test_input} (7 columns)")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions']}")

