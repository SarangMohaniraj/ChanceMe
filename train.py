import csv
import numpy as np
from matplotlib import pyplot as plt

features = None
dataset = None

with open("data/UMich.csv") as file:
	data = list(csv.reader(file))
	features = data[0]
	dataset = data[1:]

	# remove data with no sat or act score
	dataset = [i for i in filter(lambda item: item[features.index("Highest Comb SAT 1600")] != "-" or item[features.index("ACT")] != "-", dataset)]
	file.close()


with open("concordance.csv") as file:
	concordance = list(csv.reader(file))
	for element in dataset:
		sat = element[features.index("Highest Comb SAT 1600")]
		act = element[features.index("ACT")]

		# convert SAT to ACT and choose the higher score
		concordance_sat = int(concordance[int((1600-int(sat))/10)][1]) if sat != "-" else None
		act = concordance_sat if act == "-" or (concordance_sat != None and act != "-" and concordance_sat > int(act)) else int(act)
		element[features.index("ACT")] = act

		#delete SAT column because all scores have been converted to ACT
		del(element[features.index("Highest Comb SAT 1600")])

	features.remove("Highest Comb SAT 1600")
	file.close()


gpa = []
acts = []
####### IF CONVERTING TO NUMPY FLOAT32 THEN DELETE THIS ########
for element in dataset:
	#convert GPA and year to number
	element[features.index("GPA")] = float(element[features.index("GPA")])
	element[features.index("Year")] = int(element[features.index("Year")])

	gpa.append(element[features.index("GPA")])
	acts.append(element[features.index("ACT")])

plt.subplot(1,2,1)
plt.hist(gpa)
plt.subplot(1,2,2)
plt.hist(acts)
plt.show()



