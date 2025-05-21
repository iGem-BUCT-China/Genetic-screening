#读文件
file_path = r"C:\Users\86158\Desktop\retail2.dat"
with open(file_path, 'r') as file:
    old_data = file.readlines()
file_path = r"C:\Users\86158\Desktop\retail3.dat"
with open(file_path, 'r') as file:
    new_data = file.readlines()
data = old_data + new_data
#old处理
zhenmini= 0.01
#初始值
minsupport = zhenmini
c1 = {}
hang = 0
#第一次遍历计算数
for line in old_data:
    hang += 1
    items = line.strip().split()
    for item in items:
        if item in c1:
            c1[item] += 1
        else:
            c1[item] = 1
#m
minisupport = hang * minsupport
#l
L1 = {k: v for k, v in c1.items() if v >= minisupport}
#cl
def ctol(c, minisupport):
    return {k: v for k, v in c.items() if v >= minisupport}
#lc
def ltoc(L, k):
    cs = {}
    #二级遍历
    for i in range(len(L)-1):
        for j in range(i + 1, len(L)):
            #当长为2时特殊处理
            if k == 2:
                l1 = L[i:i+k-1]
                l2 = L[j:j+k-1]
                b = set(l1) | set(l2)
                c = tuple(sorted(int(x) for x in b))
                cs[c] = 0
            else:
                l1 = L[i]
                l2 = L[j]
            #进行自连接
            if k>2 and l1[:k-2] == l2[:k-2] and l1[-1] != l2[-1]:
                b = set(l1) | set(l2)
                c = tuple(sorted(int(x) for x in b))
                cs[c] = 0
    return cs
#扫描剪接非频繁项,(str(c))是否为hang的subset
def scan(data, cs):
    for line in data:
        hang = set(line.strip().split())
        for c in cs:
            if set(map(str, c)).issubset(hang):
                cs[c] += 1
    return cs

#预设项
end = [L1]
k = 2
endd = []
#循环整合频繁项集存储
while True:
    cs = ltoc(list(end[-1].keys()), k)
    if not cs:
        break
    cs = scan(old_data, cs)
    Lk = ctol(cs, minisupport)
    if not Lk:
        break
    end.append(Lk)
    k += 1

endd = []
for i in end:
    endd.extend(i.keys())

#count = len(endd)
#for i in endd:
    #print(i)
#print(count)
#cs = ltoc(list(end[-1].keys()), k)

#new处理
#初始值
minsupport = zhenmini
c1 = {}
hang = 0
#第一次遍历计算数
for line in old_data:
    hang += 1
    items = line.strip().split()
    for item in items:
        if item in c1:
            c1[item] += 1
        else:
            c1[item] = 1
#m
minisupport = hang * minsupport
#l
L1 = {k: v for k, v in c1.items() if v >= minisupport}

#预设项
new_end = [L1]
k = 2
#循环整合频繁项集存储
while True:
    cs = ltoc(list(new_end[-1].keys()), k)
    if not cs:
        break
    cs = scan(new_data, cs)
    new_Lk = ctol(cs, minisupport)
    if not Lk:
        break
    new_end.append(new_Lk)
    k += 1
new_endd = []
for i in new_end:
    new_endd.extend(i.keys())

zhen_endd=set()
new=set()
for i in endd:
    if i in new_endd:
        zhen_endd.add(i)
    else:
        a=0
        for line in data:
            items = line.strip().split()
            jihe=set(items)
            if set(map(str, i)).issubset(jihe):
                a+=1
        if a > minisupport:
            new.add(i)


for i in new_endd:
    if i not in endd:
        a=0
        for line in data:
            items = line.strip().split()
            jihe=set(items)
            if set(map(str, i)).issubset(jihe):
                a+=1
        if a > minisupport:
            new.add(i)

for i in new:
    zhen_endd.add(i)

count = 0
for i in zhen_endd:
    count+=1
    #print(i)

print(count)
