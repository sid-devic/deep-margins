import re
import sys
import matplotlib.pyplot as plt

PATH = sys.argv[1]

val_list = []
for line in open(PATH):
	if(line.find(" Validation Accuracy: ")):
		x = line.find(" Validation Accuracy:")
		val_list.append((line[x+23:x+27]))

def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

for i, s in enumerate(val_list):
	if(is_num(s)):
		if("." not in s):
			val_list[i] = "0"
			continue
		val_list[i] = float(s)
	else:
		val_list[i] = "0"
val_list = [x for x in val_list if x != "0"]

print(val_list)

plt.plot(val_list)
plt.ylabel("Validation accuracy")
plt.show()
