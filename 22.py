from collections import defaultdict

file_path = r"C:\Users\86158\Desktop\retail.dat"
with open(file_path, 'r') as file:
    hangs = [list(map(int, hang.strip().split())) for hang in file.readlines()]
minsupport = 0.0004

class TreeNode:
    def __init__(self, name, count, parent):
        self.item_name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def add(self, count):
        self.count += count

def construct_fp_tree(hangs, minsupport):
    total_hangs = len(hangs)
    minisupport = int(total_hangs * minsupport)
    count = defaultdict(int)
    for hang in hangs:
        for item in hang:
            count[item] += 1

    count = {k: v for k, v in count.items() if v >= minisupport}
    if len(count) == 0:
        return None, None
    def sort_items(hang):
        order = [item for item in hang if item in count]
        order.sort(key=lambda item: (-count[item],item))
        return order

    root = TreeNode(None, 1, None)
    header_table = defaultdict(list)
    for hang in hangs:
        order = sort_items(hang)
        current_node = root
        for item in order:
            if item in current_node.children:
                current_node.children[item].add(1)
            else:
                new_node = TreeNode(item, 1, current_node)
                current_node.children[item] = new_node
                header_table[item].append(new_node)
            current_node = current_node.children[item]
    for nodes in header_table.values():
        for i in range(1, len(nodes)):
            nodes[i - 1].node_link = nodes[i]

    return root, header_table

def mine_fp_tree(header_table, minisupport, frequent_patterns=None, prefix=[]):
    if frequent_patterns is None:
        frequent_patterns = defaultdict(list)

    def mine_node(item, node_list, base_pattern):
        conditional_patterns = []
        i = node_list[0]
        while i is not None:
            path = []
            parent = i.parent
            while parent is not None and parent.item_name is not None:
                path.append(parent.item_name)
                parent = parent.parent
            path.reverse()
            if path:
                conditional_patterns.append((path, i.count))
            i = i.node_link
            
        if not conditional_patterns:
            return
        pattern_base = []
        for pattern, count in conditional_patterns:
            for _ in range(count):
                pattern_base.append(pattern)
        conditional_tree, new_header_table = construct_fp_tree(pattern_base, minisupport / len(pattern_base))

        if conditional_tree is not None:
            conditional_tree_mine(new_header_table, minisupport, base_pattern + [item])

    def conditional_tree_mine(header_table, minisupport, base_pattern):
        order = sorted(header_table.items(), key=lambda x: len(x[1]))
        for item, nodes in order:
            new_base_pattern = base_pattern + [item]
            count = sum(node.count for node in nodes) - 1

            if count >= minisupport:
                unique_pattern = set(new_base_pattern)
                frequent_patterns[len(unique_pattern)].append((unique_pattern, count))
                mine_node(item, nodes, new_base_pattern)

    conditional_tree_mine(header_table, minisupport, prefix)
    return frequent_patterns

root, header_table = construct_fp_tree(hangs, minsupport)
if header_table:
    minisupport = int(len(hangs) * minsupport)
    frequent_itemsets = mine_fp_tree(header_table, minisupport)
print(sum(len(v) for v in frequent_itemsets.values()))