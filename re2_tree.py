import csv
import numpy as np
import time
from collections import defaultdict

class DecisionTreeNode:
    def __init__(self, column_idx=None):
        self.column_idx = column_idx  # Column to test (None for leaf)
        self.conditions = []  # List of (value_or_set, is_negated, child_node) tuples
        self.default_child = None  # For unmatched values (* or VG!:<set>)
        self.rules = []  # List of (output, specificity, rule_idx, conditions) at leaf

class RulesEngine:
    def __init__(self, csv_file):
        self.rules = None
        self.root = DecisionTreeNode()
        self.num_columns = 0
        self.condition_map = {}  # Maps normalized conditions to (output, rule_idx)
        self.max_specificity = 0
        self.load_rules(csv_file)

    def normalize_conditions(self, conditions):
        """Normalize conditions for uniqueness check."""
        normalized = []
        for c in conditions:
            if isinstance(c, tuple) and c[0] == 'not':
                normalized.append(('not', tuple(sorted(c[1]))))
            elif isinstance(c, set):
                normalized.append(tuple(sorted(c)))
            else:
                normalized.append(c)
        return tuple(normalized)

    def load_rules(self, csv_file):
        start_time = time.time()
        rules_list = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.num_columns = len(header) - 1
            for row_idx, row in enumerate(reader):
                conditions = row[:-1]
                output = row[-1]
                typed_conditions = []
                for c in conditions:
                    if c.startswith('VG!:') and c != '*':
                        typed_conditions.append(('not', set(c[4:].split(':'))))
                    elif c.startswith('VG:') and c != '*':
                        typed_conditions.append(set(c[3:].split(':')))
                    else:
                        typed_conditions.append(str(c))
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
        for rule in self.rules:
            specificity = sum(1 for c in rule[:-1] if c != '*' and not (isinstance(c, tuple) and c[0] == 'not'))
            self.max_specificity = max(self.max_specificity, specificity)
        print(f"Loaded {len(self.rules)} rules with {self.num_columns} columns, max specificity={self.max_specificity} in {time.time() - start_time:.2f}s")
        
        # Build decision tree
        start_time = time.time()
        # Heuristic: Choose column with most unique non-wildcard values
        unique_values = [defaultdict(int) for _ in range(self.num_columns)]
        for rule in self.rules:
            for i, c in enumerate(rule[:-1]):
                if c != '*' and not (isinstance(c, tuple) and c[0] == 'not'):
                    values = c if isinstance(c, set) else {c}
                    for v in values:
                        unique_values[i][v] += 1
        column_priority = sorted(range(self.num_columns), key=lambda i: len(unique_values[i]), reverse=True)
        
        def build_tree(rules_indices, depth, used_columns):
            if not rules_indices or depth >= self.num_columns:
                node = DecisionTreeNode()
                for idx in rules_indices:
                    rule = self.rules[idx]
                    conditions = rule[:-1]
                    output = rule[-1]
                    specificity = sum(1 for c in conditions if c != '*' and not (isinstance(c, tuple) and c[0] == 'not'))
                    node.rules.append((output, specificity, idx, conditions))
                    print(f"Debug: Added rule {idx} at depth {depth}: Output={output}, Specificity={specificity}")
                return node
            
            # Select next unused column with most unique values
            for col_idx in column_priority:
                if col_idx not in used_columns:
                    break
            else:
                col_idx = None
            
            node = DecisionTreeNode(column_idx=col_idx)
            if col_idx is None:
                for idx in rules_indices:
                    rule = self.rules[idx]
                    conditions = rule[:-1]
                    output = rule[-1]
                    specificity = sum(1 for c in conditions if c != '*' and not (isinstance(c, tuple) and c[0] == 'not'))
                    node.rules.append((output, specificity, idx, conditions))
                    print(f"Debug: Added rule {idx} at depth {depth}: Output={output}, Specificity={specificity}")
                return node
            
            # Group rules by column value
            value_to_indices = defaultdict(list)
            default_indices = []
            for idx in rules_indices:
                c = self.rules[idx][col_idx]
                if c == '*':
                    default_indices.append(idx)
                elif isinstance(c, tuple) and c[0] == 'not':
                    default_indices.append(idx)  # VG!:<set> uses default path
                elif isinstance(c, set):
                    for v in c:
                        value_to_indices[v].append(idx)
                else:
                    value_to_indices[c].append(idx)
            
            # Create child nodes
            for value, indices in sorted(value_to_indices.items()):
                child = build_tree(indices, depth + 1, used_columns | {col_idx})
                node.conditions.append((value, False, child))
            # Default child for * and VG!:<set>
            if default_indices:
                node.default_child = build_tree(default_indices, depth + 1, used_columns | {col_idx})
            
            node.conditions.sort(key=lambda x: x[0] or '')  # Cache-friendly sorting
            return node
        
        self.root = build_tree(list(range(len(self.rules))), 0, set())
        print(f"Built decision tree in {time.time() - start_time:.2f}s")
        
        print("\nPretty Printing Decision Tree:")
        self.pretty_print_tree(self.root)

    def pretty_print_tree(self, node, depth=0, prefix=""):
        indent = "  " * depth
        if node.column_idx is not None:
            print(f"{indent}{prefix}Column: {node.column_idx}, Rules: {len(node.rules)}")
            for value, is_negated, child in node.conditions:
                display_value = f"not:{{set}}" if is_negated else value
                print(f"{indent}  {prefix}Value: {display_value}, Negated: {is_negated}")
                self.pretty_print_tree(child, depth + 1, f"{prefix}{display_value} -> ")
            if node.default_child:
                print(f"{indent}  {prefix}Default (* or VG!:)")
                self.pretty_print_tree(node.default_child, depth + 1, f"{prefix}Default -> ")
        else:
            rule_info = [f"Rule {r[2]}: Output={r[0]}, Specificity={r[1]}" for r in node.rules]
            print(f"{indent}{prefix}Leaf, Rules: {rule_info if rule_info else 'None'}")

    def evaluate(self, input_vector, return_partial_matches=True):
        if len(input_vector) != self.num_columns:
            raise ValueError(f"Input must have {self.num_columns} values")
        
        typed_input = [str(v) for v in input_vector]
        print(f"\nDebug: Evaluating input: {typed_input}")
        
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
        partial_matches = [] if return_partial_matches else None
        
        def traverse(node, depth, current_path):
            nonlocal best_match, best_specificity, best_path, stats
            stats["nodes_visited"] += 1
            
            if node.column_idx is None or not node.conditions:
                stats["matches_found"] += len(node.rules)
                for output, specificity, idx, conditions in node.rules:
                    if specificity > best_specificity:
                        best_match = (output, specificity, idx, conditions)
                        best_specificity = specificity
                        best_path = current_path[:]
                        print(f"Debug: Found better match at depth {depth}: Rule {idx}, Output={output}, Specificity={specificity}, Path={best_path}")
                    if return_partial_matches:
                        partial_matches.append((output, specificity, idx, conditions, current_path[:]))
                return best_specificity >= self.max_specificity
            
            col_idx = node.column_idx
            value = typed_input[col_idx]
            max_possible_specificity = self.max_specificity - (depth - sum(1 for c in typed_input[:depth] if c != '*'))
            if best_specificity >= max_possible_specificity:
                print(f"Debug: Skipping depth {depth}, best_specificity={best_specificity} >= max_possible_specificity={max_possible_specificity}")
                return True
            
            print(f"Debug: At depth {depth}, column {col_idx}, checking value {value}, conditions: {[(v, neg) for v, neg, _ in node.conditions]}")
            stats["paths_explored"] += 1
            stop = False
            for condition_value, is_negated, child in node.conditions:
                if stop:
                    break
                if is_negated:
                    parent_conditions = self.rules[child.rules[0][2]][:-1] if child.rules else []
                    if col_idx < len(parent_conditions) and isinstance(parent_conditions[col_idx], tuple) and parent_conditions[col_idx][0] == 'not':
                        if value not in parent_conditions[col_idx][1]:
                            print(f"Debug: Following negated path for value {value} not in {parent_conditions[col_idx][1]}")
                            if traverse(child, depth + 1, current_path + [f"Col{col_idx}=not:{parent_conditions[col_idx][1]}"]):
                                stop = True
                elif isinstance(condition_value, set):
                    if value in condition_value:
                        print(f"Debug: Following path for value {value} in {condition_value}")
                        if traverse(child, depth + 1, current_path + [f"Col{col_idx}={value}"]):
                            stop = True
                elif condition_value == value:
                    print(f"Debug: Following path for value {value}")
                    if traverse(child, depth + 1, current_path + [f"Col{col_idx}={value}"]):
                        stop = True
            
            if node.default_child and not stop:
                stats["paths_explored"] += 1
                default_conditions = []
                for idx in range(len(self.rules)):
                    if self.rules[idx][col_idx] == '*' or (isinstance(self.rules[idx][col_idx], tuple) and self.rules[idx][col_idx][0] == 'not'):
                        default_conditions.append(self.rules[idx][col_idx])
                print(f"Debug: Following default path at column {col_idx}, conditions: {default_conditions}")
                if traverse(node.default_child, depth + 1, current_path + [f"Col{col_idx}=*"]):
                    stop = True
            
            return stop
        
        traverse(self.root, 0, [])
        stats["query_time_us"] = (time.time() - start_time) * 1_000_000
        
        if not best_match:
            print("Debug: No matching rules found")
            print("Selection Explanation: No rules matched the input. Check decision tree paths and input values.")
            print(f"Statistics: Nodes Visited={stats['nodes_visited']}, Paths Explored={stats['paths_explored']}, Matches Found={stats['matches_found']}, Query Time={stats['query_time_us']:.2f}µs")
            return None, [] if return_partial_matches else []
        
        output, specificity, idx, conditions = best_match
        print(f"Selection Explanation:")
        print(f"  Best Match: Rule {idx}, Output={output}, Specificity={specificity}")
        print(f"  Conditions: {[list(c[1]) if isinstance(c, tuple) and c[0] == 'not' else list(c) if isinstance(c, set) else c for c in conditions.tolist()]}")
        print(f"  Path Taken: {best_path}")
        print(f"  Why Selected: This rule has the highest specificity ({specificity}) among {stats['matches_found']} matching rules.")
        
        if return_partial_matches:
            partial_matches = [
                {
                    "output": output,
                    "specificity": specificity,
                    "rule_index": idx,
                    "conditions": [list(c[1]) if isinstance(c, tuple) and c[0] == 'not' else list(c) if isinstance(c, set) else c for c in conditions.tolist()],
                    "path": path
                }
                for output, specificity, idx, conditions, path in partial_matches
            ]
        else:
            partial_matches = []
        
        print(f"Statistics: Nodes Visited={stats['nodes_visited']}, Paths Explored={stats['paths_explored']}, Matches Found={stats['matches_found']}, Query Time={stats['query_time_us']:.2f}µs")
        return output, partial_matches

# Example usage
if __name__ == "__main__":
    # Simulate CSV
    csv_content = """Client,LE,Flag1,Trader_Domicile,Flag2,Output,Group1,Group2
*,NA,1,*,*,"A|B|C",Stocks,Equities
001,NA,1,*,*,"A:soft|B|C",Stocks,Equities
*,VG:EUR:PUBLIC,*,EU,*,"C|D",Stocks,Equities
*,VG!:EUR:PUBLIC,*,GB,*,"C:hard|D",Stocks,Equities"""
    with open("rules.csv", "w") as f:
        f.write(csv_content)

    # Test duplicate/conflict
    try:
        csv_content_duplicate = """Client,LE,Flag1,Trader_Domicile,Flag2,Output,Group1,Group2
*,NA,1,*,*,"A|B|C",Stocks,Equities
*,NA,1,*,*,"A|B|C",Stocks,Equities"""
        with open("rules_duplicate.csv", "w") as f:
            f.write(csv_content_duplicate)
        engine = RulesEngine("rules_duplicate.csv")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        csv_content_conflict = """Client,LE,Flag1,Trader_Domicile,Flag2,Output,Group1,Group2
*,NA,1,*,*,"A|B|C",Stocks,Equities
*,NA,1,*,*,"C|D",Stocks,Equities"""
        with open("rules_conflict.csv", "w") as f:
            f.write(csv_content_conflict)
        engine = RulesEngine("rules_conflict.csv")
    except ValueError as e:
        print(f"Error: {e}")

    # Initialize engine
    engine = RulesEngine("rules.csv")

    # Test case 1
    test_input = ["001", "NA", "1", "EU", "0", "Stocks", "Equities"]
    best_output, partial_matches = engine.evaluate(test_input, return_partial_matches=True)
    print(f"\nInput: {test_input} (7 columns)")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions']}, Path={match['path']}")

    # Test case 2
    test_input = ["002", "EUR", "0", "EU", "0", "Stocks", "Equities"]
    best_output, partial_matches = engine.evaluate(test_input, return_partial_matches=True)
    print(f"\nInput: {test_input} (7 columns)")
    print(f"Best Match Output: {best_output}")
    print("Partial Matches:")
    for match in partial_matches:
        print(f"  Rule {match['rule_index']}: Output={match['output']}, Specificity={match['specificity']}, Conditions={match['conditions']}, Path={match['path']}")