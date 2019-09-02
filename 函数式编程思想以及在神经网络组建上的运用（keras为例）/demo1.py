





abc = ['com','fnng','cnblogs']
for i in range(len(abc)):
    print(len(abc[i]))



abc_len = map(len,['hao','fnng','cnblogs'])
print(list(abc_len))







#大小写转换
abc = ['cOm','FNng','cnBLoGs']
lowname = []
for i in range(len(abc)):
    lowname.append(abc[i].lower())
print(list(lowname))



def to_lower(item):
    return item.lower()
name = map(to_lower,['cOm','FNng','cnBLoGs'])
print(list(name))
#========输出===========
# ['com', 'fnng', 'cnblogs']


from functools import reduce
def add(a,b):
    return a+b
add = reduce(add,[2,3,4])
print(add)








