import csv
import numpy as np
import time
from collections import defaultdict

class DecisionTreeNode:
    def __init__(self, column_idx=None):
        self.column_idx = column_idx
        self.conditions = []  # (value_or_set, is_negated, child_node)
        self.default_child = None
        self.rules = []  # (outputs_tuple, specificity, rule_idx, conditions)

class RulesEngine:
    def __init__(self, csv_files):
        self.rules = None
        self.root = DecisionTreeNode()
        self.num_columns = 0
        self.column_map = {}
        self.condition_map = {}
        self.max_specificity = 0
        self.output_names = []
        self.load_rules(csv_files)

    def normalize_conditions(self, conditions):
        normalized = []
        for c in conditions:
            if isinstance(c, str) and c.startswith('VG:') and c != '*':
                normalized.append(set(c[3:].split(':')))
            else:
                normalized.append(c)
        return tuple(normalized)

    def load_rules(self, csv_files):
        start_time = time.time()
        rules_list = []
        all_columns = set()
        output_names = set()

        for csv_file in csv_files:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                output_names.add(header[-1])
                for col in header[:-1]:
                    all_columns.add(col)
        
        all_columns = sorted(list(all_columns))
        self.num_columns = len(all_columns)
        if self.num_columns > 128:
            raise ValueError(f"Number of columns ({self.num_columns}) exceeds maximum limit of 128")
        self.column_map = {col: idx for idx, col in enumerate(all_columns)}
        self.output_names = sorted(list(output_names))

        rule_idx = 0
        for csv_file in csv_files:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                file_output = header[-1]
                file_columns = header[:-1]
                for row in reader:
                    output = row[-1]
                    conditions = row[:-1]
                    full_conditions = ['*'] * self.num_columns
                    for col, val in zip(file_columns, conditions):
                        if col in self.column_map:
                            full_conditions[self.column_map[col]] = val
                    typed_conditions = []
                    for c in full_conditions:
                        if isinstance(c, str) and c.startswith('VG:') and c != '*':
                            typed_conditions.append(set(c[3:].split(':')))
                        else:
                            typed_conditions.append(c)
                    outputs_tuple = [None] * len(self.output_names)
                    output_idx = self.output_names.index(file_output)
                    outputs_tuple[output_idx] = output
                    normalized_conditions = self.normalize_conditions(typed_conditions)
                    if normalized_conditions in self.condition_map:
                        prev_outputs, prev_idx = self.condition_map[normalized_conditions]
                        if prev_outputs == tuple(outputs_tuple):
                            raise ValueError(
                                f"Duplicate rule at row {rule_idx + 2} in {csv_file}: "
                                f"Conditions={typed_conditions}, Outputs={outputs_tuple}. "
                                f"Matches rule {prev_idx + 2}."
                            )
                        else:
                            raise ValueError(
                                f"Conflicting rule at row {rule_idx + 2} in {csv_file}: "
                                f"Conditions={typed_conditions}, Outputs={outputs_tuple}. "
                                f"Matches rule {prev_idx + 2} with outputs={prev_outputs}."
                            )
                    self.condition_map[normalized_conditions] = (outputs_tuple, rule_idx)
                    rules_list.append(typed_conditions + [outputs_tuple])
                    rule_idx += 1
        
        self.rules = np.array(rules_list, dtype=object)
        for rule in self.rules:
            specificity = sum(1 for c in rule[:-1] if c != '*' and not isinstance(c, set))
            self.max_specificity = max(self.max_specificity, specificity)
        print(f"Loaded {len(self.rules)} rules with {self.num_columns} columns, max specificity={self.max_specificity} in {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        unique_values = [defaultdict(int) for _ in range(self.num_columns)]
        for rule in self.rules:
            for i, c in enumerate(rule[:-1]):
                if c != '*' and not isinstance(c, set):
                    unique_values[i][c] += 1
                elif isinstance(c, set):
                    for v in c:
                        unique_values[i][v] += 1
        column_priority = sorted(range(self.num_columns), key=lambda i: len(unique_values[i]), reverse=True)
        
        self.root = self.build_tree(list(range(len(self.rules))), 0, set())
        print(f"Built decision tree in {time.time() - start_time:.2f}s")
        
        print("\nPretty Printing Decision Tree:")
        self.pretty_print_tree(self.root)

    def build_tree(self, rules_indices, depth, used_columns):
        if not rules_indices or depth >= self.num_columns:
            node = DecisionTreeNode()
            for idx in rules_indices:
                rule = self.rules[idx]
                conditions = rule[:-1]
                outputs = rule[-1]
                specificity = sum(1 for c in conditions if c != '*' and not isinstance(c, set))
                node.rules.append((outputs, specificity, idx, conditions))
            return node
        
        node = DecisionTreeNode(column_idx=column_priority[depth] if depth < len(column_priority) else None)
        if node.column_idx is None:
            for idx in rules_indices:
                rule = self.rules[idx]
                conditions = rule[:-1]
                outputs = rule[-1]
                specificity = sum(1 for c in conditions if c != '*' and not isinstance(c, set))
                node.rules.append((outputs, specificity, idx, conditions))
            return node
        
        value_to_indices = defaultdict(list)
        default_indices = []
        for idx in rules_indices:
            c = self.rules[idx][node.column_idx]
            if c == '*':
                default_indices.append(idx)
            elif isinstance(c, set):
                for v in c:
                    value_to_indices[v].append(idx)
            else:
                value_to_indices[c].append(idx)
        
        for value, indices in sorted(value_to_indices.items(), key=lambda x: str(x[0])):
            if indices:  # Only create nodes for non-empty indices
                child = self.build_tree(indices, depth + 1, used_columns | {node.column_idx})
                if isinstance(value, set):
                    node.conditions.append((value, False, child))
                else:
                    node.conditions.append((value, False, child))
        if default_indices:
            node.default_child = self.build_tree(default_indices, depth + 1, used_columns | {node.column_idx})
        
        return node

    def pretty_print_tree(self, node, depth=0, prefix=""):
        indent = "  " * depth
        if node.column_idx is not None:
            print(f"{indent}{prefix}Column: {node.column_idx}, Rules: {len(node.rules)}")
            for value, is_negated, child in node.conditions:
                display_value = f"{{{','.join(sorted(value))}}}" if isinstance(value, set) else str(value)
                print(f"{indent}  {prefix}Value: {display_value}, Negated: {is_negated}")
                self.pretty_print_tree(child, depth + 1, f"{prefix}{display_value} -> ")
            if node.default_child:
                print(f"{indent}  {prefix}Default (*)")
                self.pretty_print_tree(node.default_child, depth + 1, f"{prefix}Default -> ")
        else:
            rule_info = [f"Rule {r[2]}: Outputs={r[0]}, Specificity={r[1]}, Conditions={r[3]}" for r in node.rules]
            print(f"{indent}{prefix}Leaf, Rules: {rule_info if rule_info else 'None'}")

    def evaluate(self, input_vector, return_partial_matches=True):
        input_len = len(input_vector)
        if input_len > 128:
            raise ValueError(f"Input length ({input_len}) exceeds maximum limit of 128")
        typed_input = [str(v) for v in input_vector] + ['*'] * (self.num_columns - input_len)
        print(f"\nDebug: Evaluating input: {typed_input} (original: {input_vector})")
        
        start_time = time.time()
        stats = {
            "nodes_visited": 0,
            "paths_explored": 0,
            "matches_found": 0,
            "query_time_us": 0
        }
        best_match = None
        best_specificity = -1
        best_path = []
        best_rule_idx = float('inf')
        partial_matches = [] if return_partial_matches else None
        
        def matches_rule(conditions, input_vector):
            for i, (c, v) in enumerate(zip(conditions, input_vector)):
                if c == '*':
                    continue
                elif isinstance(c, set):
                    if v not in c:
                        return False
                elif c != v:
                    return False
            return True

        def traverse(node, depth, current_path):
            nonlocal best_match, best_specificity, best_path, best_rule_idx, stats
            stats["nodes_visited"] += 1
            
            if node.column_idx is None or not node.conditions:
                for outputs, specificity, idx, conditions in node.rules:
                    if matches_rule(conditions, typed_input):
                        stats["matches_found"] += 1
                        if specificity > best_specificity or (specificity == best_specificity and idx < best_rule_idx):
                            best_match = (outputs, specificity, idx, conditions)
                            best_specificity = specificity
                            best_path = current_path[:]
                            best_rule_idx = idx
                            print(f"Debug: Found better match at depth {depth}: Rule {idx}, Outputs={outputs}, Specificity={specificity}, Path={best_path}")
                        if return_partial_matches:
                            partial_matches.append((outputs, specificity, idx, conditions, current_path[:]))
                return
            
            col_idx = node.column_idx
            value = typed_input[col_idx]
            print(f"Debug: At depth {depth}, column {col_idx}, checking value {value}, conditions: {[(v, neg) for v, neg, _ in node.conditions]}")
            stats["paths_explored"] += 1
            
            for condition_value, is_negated, child in node.conditions:
                if isinstance(condition_value, set):
                    if value in condition_value:
                        print(f"Debug: Following path for value {value} in {condition_value}")
                        traverse(child, depth + 1, current_path + [f"Col{col_idx}={value}"])
                elif condition_value == value:
                    print(f"Debug: Following path for value {value}")
                    traverse(child, depth + 1, current_path + [f"Col{col_idx}={value}"])
            
            if node.default_child:
                stats["paths_explored"] += 1
                print(f"Debug: Following default path at column {col_idx}, conditions: ['*']")
                traverse(node.default_child, depth + 1, current_path + [f"Col{col_idx}=*"])
        
        traverse(self.root, 0, [])
        stats["query_time_us"] = (time.time() - start_time) * 1_000_000
        
        if not best_match:
            print("Debug: No matching rules found")
            print("Selection Explanation: No rules matched the input. Check decision tree paths and input values.")
            print(f"Statistics: Nodes Visited={stats['nodes_visited']}, Paths Explored={stats['paths_explored']}, Matches Found={stats['matches_found']}, Query Time={stats['query_time_us']:.2f}µs")
            return None, [] if return_partial_matches else []
        
        outputs, specificity, idx, conditions = best_match
        print(f"Selection Explanation:")
        print(f"  Best Match: Rule {idx}, Outputs={outputs}, Specificity={specificity}")
        print(f"  Conditions: {[list(c) if isinstance(c, set) else c for c in conditions]}")
        print(f"  Path Taken: {best_path}")
        print(f"  Why Selected: This rule has the highest specificity ({specificity}) among {stats['matches_found']} matching rules (tie broken by lowest rule index {idx}).")
        
        if return_partial_matches:
            partial_matches = [
                {
                    "outputs": outputs,
                    "specificity": specificity,
                    "rule_index": idx,
                    "conditions": [list(c) if isinstance(c, set) else c for c in conditions],
                    "path": path
                }
                for outputs, specificity, idx, conditions, path in sorted(partial_matches, key=lambda x: (-x[1], x[2]))
            ]
        else:
            partial_matches = []
        
        print(f"Statistics: Nodes Visited={stats['nodes_visited']}, Paths Explored={stats['paths_explored']}, Matches Found={stats['matches_found']}, Query Time={stats['query_time_us']:.2f}µs")
        return outputs, partial_matches

# Example usage
if __name__ == "__main__":
    # Simulate CSV file with 11 rules, 2 columns, VG: and *
    csv_content = """A,B,Output1
1,VG:EUR:PUBLIC,o1
VG:USA:UK,2,o2
VG:FR:DE,*,o3
VG:IT:ES,3,o4
VG:CH:JP,*,o5
1,4,o6
VG:AU:NZ,5,o7
2,VG:BR:AR,o8
VG:CA:MX,*,o9
3,VG:IN:SG,o10
VG:RU:KR,2,o11"""
    with open("rules.csv", "w") as f:
        f.write(csv_content)

    # Initialize engine
    engine = RulesEngine(["rules.csv"])

    # Test inputs
    test_inputs = [
        ["1", "EUR"], ["USA", "2"], ["FR", "*"], ["IT", "3"], ["CH", "*"],
        ["1", "4"], ["AU", "5"], ["2", "BR"], ["CA", "*"], ["3", "IN"], ["RU", "2"]
    ]
    for test_input in test_inputs:
        best_outputs, partial_matches = engine.evaluate(test_input, return_partial_matches=True)
        print(f"\nInput: {test_input} (2 columns)")
        print(f"Best Match Outputs: {best_outputs}")
        print("Partial Matches:")
        for match in partial_matches:
            print(f"  Rule {match['rule_index']}: Outputs={match['outputs']}, Specificity={match['specificity']}, Conditions={match['conditions']}, Path={match['path']}")
