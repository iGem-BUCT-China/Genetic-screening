# 读取文件
file_path = r"C:\Users\86158\Desktop\retail.dat"
with open(file_path, 'r') as file:
    data = file.readlines()

# 初始化参数
min_support = 0.01
min_confidence = 0.5
c1 = {}
transactions = []
transaction_count = 0

# 数据预处理
for line in data:
    transaction_count += 1
    items = line.strip().split()
    transactions.append(set(items))
    for item in items:
        c1[item] = c1.get(item, 0) + 1

# 计算最小支持数
min_support_count = transaction_count * min_support

# 生成初始频繁1项集
L1 = {frozenset([k]): v for k, v in c1.items() if v >= min_support_count}

# 剪枝函数
def prune_candidates(candidates, min_support_count):
    return {k: v for k, v in candidates.items() if v >= min_support_count}

# 生成候选项集
def generate_candidates(prev_level, k):
    candidates = {}
    items = list(prev_level)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            candidate = items[i] | items[j]
            if len(candidate) == k:
                candidates[frozenset(candidate)] = 0
    return candidates

# 扫描事务，计算候选项集的支持度
def scan_transactions(transactions, candidates):
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                candidates[candidate] += 1
    return candidates

# 挖掘频繁项集
frequent_itemsets = [L1]
k = 2

while True:
    candidates = generate_candidates(frequent_itemsets[-1], k)
    if not candidates:
        break
    candidates = scan_transactions(transactions, candidates)
    Lk = prune_candidates(candidates, min_support_count)
    if not Lk:
        break
    frequent_itemsets.append(Lk)
    k += 1

# 关联规则挖掘
rules = []

for level in frequent_itemsets[1:]:
    for itemset, support_count in level.items():
        for antecedent in map(frozenset, [set(itemset) - {i} for i in itemset]):
            consequent = itemset - antecedent
            if antecedent and consequent:
                antecedent_support = sum(1 for t in transactions if antecedent.issubset(t))
                confidence = support_count / antecedent_support
                if confidence >= min_confidence:
                    support = support_count / transaction_count
                    rules.append((antecedent, consequent, support, confidence))

# 输出频繁项集
print("频繁项集：")
for level in frequent_itemsets:
    for itemset, support_count in level.items():
        print(f"{set(itemset)}: 支持度={support_count / transaction_count:.2f}")

# 输出关联规则
print("\n关联规则:")
for antecedent, consequent, support, confidence in rules:
    print(f"{set(antecedent)} => {set(consequent)} (支持度={support:.2f}, 置信度={confidence:.2f})")

