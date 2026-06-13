#!/usr/bin/env python3

"""
Rule Conflict Detector

Supports:
- Wildcards (empty cell -> *)
- Value Groups:
      A:B:C@valuegroup

CSV Format:

M1,M2,M3,|,Output
A:B:C@valuegroup,,,|,99
,B,,|,100
X,Y,Z,|,200

Meaning:

Rule1:
    M1 in {A,B,C}
    M2=*
    M3=*
    Output=99

Rule2:
    M1=*
    M2=B
    M3=*
    Output=100

Conflict definition:
    Two rules conflict if:
      - Their input spaces overlap
      - Outputs differ

This implementation:
    - Uses an inverted index for pruning
    - Builds an overlap graph
    - Finds connected conflict clusters
    - Reports witness values showing why rules overlap
"""

import csv
from collections import defaultdict, deque

WILDCARD = "*"


# ============================================================
# Rule
# ============================================================

class Rule:
    def __init__(self, rule_id, conditions, output):
        self.id = rule_id
        self.conditions = conditions
        self.output = output

    def __repr__(self):
        return f"{self.id}: {self.conditions} -> {self.output}"


# ============================================================
# CSV Parsing
# ============================================================

def parse_cell(cell):
    """
    Empty => wildcard

    A:B:C@valuegroup
      =>
    {"VG": {"A","B","C"}}
    """

    cell = cell.strip()

    if not cell:
        return WILDCARD

    lower = cell.lower()

    if "@valuegroup" in lower:

        values_part = cell.split("@")[0]

        values = {
            v.strip()
            for v in values_part.split(":")
            if v.strip()
        }

        return {"VG": values}

    return cell


def load_ruleset_csv(filename):
    rules = []

    with open(filename, newline="", encoding="utf-8") as f:

        reader = csv.reader(f)

        header = next(reader)

        try:
            pipe_index = header.index("|")
        except ValueError:
            raise ValueError(
                "Header must contain a '|' column separating inputs and output"
            )

        input_columns = header[:pipe_index]

        output_index = pipe_index + 1

        for row_num, row in enumerate(reader, start=1):

            if not row:
                continue

            if len(row) <= output_index:
                raise ValueError(
                    f"Row {row_num} missing output column"
                )

            conditions = {}

            for idx, attr in enumerate(input_columns):
                value = row[idx] if idx < len(row) else ""
                conditions[attr] = parse_cell(value)

            output = row[output_index].strip()

            rules.append(
                Rule(
                    rule_id=f"R{row_num}",
                    conditions=conditions,
                    output=output,
                )
            )

    return rules


# ============================================================
# Conflict Engine
# ============================================================

class ConflictEngine:

    def __init__(self):
        self.rules = []

        # attr -> value -> set(rule_ids)
        self.index = defaultdict(lambda: defaultdict(set))

    # --------------------------------------------------------
    # Normalize values
    # --------------------------------------------------------

    def expand(self, value):

        if value == WILDCARD:
            return None

        if isinstance(value, dict) and "VG" in value:
            return set(value["VG"])

        return {value}

    # --------------------------------------------------------
    # Add rule
    # --------------------------------------------------------

    def add_rule(self, rule):

        self.rules.append(rule)

        for attr, value in rule.conditions.items():

            values = self.expand(value)

            if values is None:
                continue

            for v in values:
                self.index[attr][v].add(rule.id)

    # --------------------------------------------------------
    # Candidate pruning
    # --------------------------------------------------------

    def candidates(self, rule):

        candidate_sets = []

        for attr, value in rule.conditions.items():

            values = self.expand(value)

            if values is None:
                continue

            current = set()

            for v in values:
                current |= self.index[attr].get(v, set())

            candidate_sets.append(current)

        if not candidate_sets:
            return {r.id for r in self.rules}

        result = candidate_sets[0]

        for s in candidate_sets[1:]:
            result &= s

        return result

    # --------------------------------------------------------
    # Overlap test
    # --------------------------------------------------------

    def overlaps(self, r1, r2):

        attrs = set(r1.conditions) | set(r2.conditions)

        for attr in attrs:

            v1 = self.expand(
                r1.conditions.get(attr, WILDCARD)
            )

            v2 = self.expand(
                r2.conditions.get(attr, WILDCARD)
            )

            # wildcard
            if v1 is None or v2 is None:
                continue

            if v1.isdisjoint(v2):
                return False

        return True

    # --------------------------------------------------------
    # Witness
    # --------------------------------------------------------

    def witness(self, r1, r2):
        """
        Produce one example input that matches both rules.
        """

        attrs = set(r1.conditions) | set(r2.conditions)

        witness = {}

        for attr in sorted(attrs):

            v1 = self.expand(
                r1.conditions.get(attr, WILDCARD)
            )

            v2 = self.expand(
                r2.conditions.get(attr, WILDCARD)
            )

            if v1 is None and v2 is None:
                witness[attr] = "*"
                continue

            if v1 is None:
                witness[attr] = next(iter(v2))
                continue

            if v2 is None:
                witness[attr] = next(iter(v1))
                continue

            common = v1 & v2

            if not common:
                return None

            witness[attr] = next(iter(common))

        return witness

    # --------------------------------------------------------
    # Build overlap graph
    # --------------------------------------------------------

    def build_graph(self):

        graph = defaultdict(set)

        id_map = {
            r.id: r
            for r in self.rules
        }

        for rule in self.rules:

            for candidate_id in self.candidates(rule):

                if candidate_id == rule.id:
                    continue

                other = id_map[candidate_id]

                if not self.overlaps(rule, other):
                    continue

                graph[rule.id].add(candidate_id)
                graph[candidate_id].add(rule.id)

        return graph

    # --------------------------------------------------------
    # Detect conflicts
    # --------------------------------------------------------

    def detect_conflicts(self):

        graph = self.build_graph()

        id_map = {
            r.id: r
            for r in self.rules
        }

        visited = set()

        conflict_clusters = []

        for node in graph:

            if node in visited:
                continue

            queue = deque([node])

            cluster_ids = []

            while queue:

                current = queue.popleft()

                if current in visited:
                    continue

                visited.add(current)
                cluster_ids.append(current)

                for nxt in graph[current]:
                    if nxt not in visited:
                        queue.append(nxt)

            cluster_rules = [
                id_map[rid]
                for rid in cluster_ids
            ]

            outputs = {
                r.output
                for r in cluster_rules
            }

            if len(outputs) > 1:
                conflict_clusters.append(cluster_rules)

        return conflict_clusters


# ============================================================
# Reporting
# ============================================================

def print_conflicts(engine, clusters):

    if not clusters:
        print("\nNo conflicts detected.")
        return

    print("\n=== CONFLICT CLUSTERS ===")

    for idx, cluster in enumerate(clusters, start=1):

        print(f"\nCluster #{idx}")

        for r in cluster:
            print(
                f"  {r.id}: {r.conditions} -> {r.output}"
            )

        print("\n  Pairwise overlap witnesses:")

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):

                r1 = cluster[i]
                r2 = cluster[j]

                if not engine.overlaps(r1, r2):
                    continue

                witness = engine.witness(r1, r2)

                if witness:

                    print(
                        f"\n    {r1.id} <-> {r2.id}"
                    )

                    print(
                        f"      outputs: "
                        f"{r1.output} vs {r2.output}"
                    )

                    print(
                        f"      witness: {witness}"
                    )


# ============================================================
# Main
# ============================================================

def main():

    filename = "rules.csv"

    rules = load_ruleset_csv(filename)

    print("\n=== LOADED RULES ===\n")

    for r in rules:
        print(r)

    engine = ConflictEngine()

    for r in rules:
        engine.add_rule(r)

    conflicts = engine.detect_conflicts()

    print_conflicts(engine, conflicts)


if __name__ == "__main__":
    main()
