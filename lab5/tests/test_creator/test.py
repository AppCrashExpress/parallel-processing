import random

size = 513
vals = [-i + size // 2 for i in range(size)]

vals = random.choices(vals, k=size)

print(size)
print( ' '.join(map(str, vals)) )
