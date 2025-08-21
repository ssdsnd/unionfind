import csv
import numpy as np
import time
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = []  # List of (value, is_negated, child_node) tuples
        self.rules = []  # List of (output, specificity, rule_idx, conditions)

class RulesEngine:
    def __init__(self, csv_file):
        self.rules = None
        self.trie = TrieNode()
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
            conditions = rule[:-1]
            specificity = sum(1 for c in conditions if c != '*' and not (isinstance(c, tuple) and c[0] == 'not'))
            self.max_specificity = max(self.max_specificity, specificity)
        print(f"Loaded {len(self.rules)} rules with {self.num_columns} columns, max specificity={self.max_specificity} in {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        for idx, rule in enumerate(self.rules):
            conditions = rule[:-1]
            output = rule[-1]
            specificity = sum(1 for c in conditions if c != '*' and not (isinstance(c, tuple) and c[0] == 'not'))
            def build_path(node, depth, path_conditions):
                if depth == self.num_columns:
                    node.rules.append((output, specificity, idx, conditions))
                    print(f"Debug: Added rule {idx} at depth {depth}: {path_conditions}, Output={output}, Specificity={specificity}")
                    return
                c = conditions[depth]
                if isinstance(c, tuple) and c[0] == 'not':
                    child = next((n for v, neg, n in node.children if v is None and neg), None)
                    if not child:
                        child = TrieNode()
                        node.children.append((None, True, child))
                        node.children.sort(key=lambda x: (x[1], x[0] or ''))
                    build_path(child, depth + 1, path_conditions + [f"not:{c[1]}"])
                elif isinstance(c, set):
                    for v in sorted(c):
                        child = next((n for v_, neg, n in node.children if v_ == v and not neg), None)
                        if not child:
                            child = TrieNode()
                            node.children.append((v, False, child))
                            node.children.sort(key=lambda x: (x[1], x[0] or ''))
                        build_path(child, depth + 1, path_conditions + [v])
                else:
                    child = next((n for v_, neg, n in node.children if v_ == c and not neg), None)
                    if not child:
                        child = TrieNode()
                        node.children.append((c, False, child))
                        node.children.sort(key=lambda x: (x[1], x[0] or ''))
                    build_path(child, depth + 1, path_conditions + [c])
            
            build_path(self.trie, 0, [])
        print(f"Built trie in {time.time() - start_time:.2f}s")
        
        print("\nPretty Printing Trie:")
        self.pretty_print_trie(self.trie)

    def pretty_print_trie(self, node, depth=0, prefix=""):
        indent = "  " * depth
        for value, is_negated, child in node.children:
            display_value = f"not:{{set}}" if is_negated else value
            rule_info = [f"Rule {r[2]}: Output={r[0]}, Specificity={r[1]}" for r in child.rules]
            print(f"{indent}{prefix}Value: {display_value}, Negated: {is_negated}, Rules: {rule_info if rule_info else 'None'}")
            self.pretty_print_trie(child, depth + 1, f"{prefix}{display_value} -> ")

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
            
            if depth == self.num_columns:
                for output, specificity, idx, conditions in node.rules:
                    stats["matches_found"] += 1
                    if specificity > best_specificity:
                        best_match = (output, specificity, idx, conditions)
                        best_specificity = specificity
                        best_path = current_path[:]
                        print(f"Debug: Found better match at depth {depth}: Rule {idx}, Output={output}, Specificity={specificity}, Path={best_path}")
                    if return_partial_matches:
                        partial_matches.append((output, specificity, idx, conditions, current_path[:]))
                return best_specificity >= self.max_specificity
            
            value = typed_input[depth]
            max_possible_specificity = self.max_specificity - (depth - sum(1 for c in typed_input[:depth] if c != '*'))
            if best_specificity >= max_possible_specificity:
                print(f"Debug: Skipping depth {depth}, best_specificity={best_specificity} >= max_possible_specificity={max_possible_specificity}")
                return True
            
            stop = False
            print(f"Debug: At depth {depth}, checking value {value}, available children: {[(v, neg) for v, neg, _ in node.children]}")
            for child_value, is_negated, child in node.children:
                if stop:
                    break
                stats["paths_explored"] += 1
                if is_negated:
                    parent_conditions = self.rules[child.rules[0][2]][:-1] if child.rules else []
                    if depth < len(parent_conditions) and isinstance(parent_conditions[depth], tuple) and parent_conditions[depth][0] == 'not':
                        if value not in parent_conditions[depth][1]:
                            print(f"Debug: Following negated path for value {value} not in {parent_conditions[depth][1]}")
                            if traverse(child, depth + 1, current_path + [f"not:{parent_conditions[depth][1]}"]):
                                stop = True
                elif child_value == value or child_value == '*':
                    print(f"Debug: Following path for value {child_value}")
                    if traverse(child, depth + 1, current_path + [child_value]):
                        stop = True
            
            return stop
        
        traverse(self.trie, 0, [])
        stats["query_time_us"] = (time.time() - start_time) * 1_000_000
        
        if not best_match:
            print("Debug: No matching rules found")
            print("Selection Explanation: No rules matched the input. Check Trie paths and input values.")
            print(f"Statistics: Nodes Visited={stats['nodes_visited']}, Paths Explored={stats['paths_explored']}, Matches Found={stats['matches_found']}, Query Time={stats['query_time_us']:.2f}µs")
            return None, [] if return_partial_matches else []
        
        # Explain best match
        output, specificity, idx, conditions = best_match
        print(f"Selection Explanation:")
        print(f"  Best Match: Rule {idx}, Output={output}, Specificity={specificity}")
        print(f"  Conditions: {[list(c[1]) if isinstance(c, tuple) and c[0] == 'not' else list(c) if isinstance(c, set) else c for c in conditions.tolist()]}")
        print(f"  Path Taken: {best_path}")
        print(f"  Why Selected: This rule has the highest specificity ({specificity}) among {stats['matches_found']} matching rules.")
        
        # Format partial matches
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