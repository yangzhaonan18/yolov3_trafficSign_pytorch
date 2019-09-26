import os 

dirs = os.listdir("../ALL_IN")
count = 0
for d in dirs:
	
	if not d.endswith(".py"):
		names = os.listdir(d)
		print(d, len(names))
		count += len(names)
print(count)

