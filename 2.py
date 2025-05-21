#定义FP树的类
class tree:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.Link = None
        self.parent = parent
        self.children = {}
    #加法
    def add(self, num):
        self.count += num
#创建FP新树
def createTree(data, minSup=1):
    head = {}
    for hang in data:
        for i in hang:
            head[i] = head.get(i, 0) + data[hang]
    head = {k: v for k, v in head.items() if v >= minSup}
    #创根树、链表
    pinfan = set(head.keys())
    if len(pinfan) == 0: return None, None
    for k in head:
        head[k] = [head[k], None]
    Tree = tree('Null', 1, None)
    #插入新数据不断延伸
    for hangs, count in data.items():
        a = {}
        for i in hangs:
            if i in pinfan:
                a[i] = head[i][0]
        if len(a) > 0:
            order = [v[0] for v in sorted(a.items(), key=lambda p: p[1], reverse=True)]
            updateTree(order, Tree, head, count)
    return Tree, head
#更新树
def updateTree(items, inTree, head, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].add(count)
    #若为新点、链表更新
    else:
        inTree.children[items[0]] = tree(items[0], count, inTree)
        if head[items[0]][1] == None:
            head[items[0]][1] = inTree.children[items[0]]
        else:
            updateHead(head[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], head, count)

#更新FP
def updateHead(last, new):
    while (last.Link != None):
        last = last.Link
    last.Link = new
#向上遍历
def ascendTree(now, road):
    if now.parent != None:
        road.append(now.name)
        ascendTree(now.parent, road)
#结合
def findroad(basePat, treeNode):
    road = {}
    #汇总条件模式基
    while treeNode != None:
        preroad = []
        ascendTree(treeNode, preroad)
        if len(preroad) > 1:
            road[frozenset(preroad[1:])] = treeNode.count
        treeNode = treeNode.Link
    return road
#fp
def mineTree(inTree, head, minSup, preFix, freList):
    order = [v[0] for v in sorted(head.items(), key=lambda p: p[1][0])]
    #堆条件模式基
    for i in order:
        newFre = preFix.copy()
        newFre.add(i)
        freList.append(newFre)
        road = findroad(i, head[i][1])
        newTree, Head = createTree(road, minSup)
#递归
        if Head != None:
            mineTree(newTree, Head, minSup, newFre, freList)
#读文件
file_path = r"C:\Users\86158\Desktop\retail.dat"
with open(file_path, 'r') as file:
    data = file.readlines()
#初始定义
minsupport = 0.009
hang = 0
c1 = {}
sdata = []
#第一次遍历计算数
for line in data:
    hang += 1
    items = line.strip().split()
    for item in items:
        if item in c1:
            c1[item] += 1
        else:
            c1[item] = 1
#m
minisupport = hang * minsupport
#找频繁
L1 = {k: v for k, v in c1.items() if v >= minisupport}
L1 = dict(sorted(L1.items(), key=lambda item: item[1], reverse=True))
keys = list(L1.keys())
#保留频繁
for line in data:
    items = line.strip().split()
    hang = [item for item in keys if item in items]
    sdata.append(frozenset(hang))
#记录出现次数
dataSet = {}
for transaction in sdata:
    if transaction in dataSet:
        dataSet[transaction] += 1
    else:
        dataSet[transaction] = 1

# 构
fpTree, head = createTree(dataSet, minisupport)

# 挖掘
freList = []
mineTree(fpTree, head, minisupport, set([]), freList)

# 输出频繁项

count=len(freList)
for itemset in freList:
    print(itemset)
print(count)