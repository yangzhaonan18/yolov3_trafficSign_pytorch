import os 

dirs = os.listdir("../all_crop_data/")
count = 0
nums = []
for d in dirs:
	
	if not d.endswith(".py"):
		names = os.listdir(d)
		# print(len(names))
		count += len(names)
		nums.append([len(names), d])
		nums = sorted(nums, reverse=True)
for i in nums:
	print(i)
print(count)

