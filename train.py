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


gpa = []
acts = []

with open("concordance.csv") as file:
	concordance = list(csv.reader(file))
	for element in dataset:
		sat = element[features.index("Highest Comb SAT 1600")]
		act = element[features.index("ACT")]

		# convert SAT to ACT and choose the higher score
		concordance_sat = int(concordance[int((1600-int(sat))/10)][1]) if sat != "-" else None
		act = concordance_sat if act == "-" or (concordance_sat != None and act != "-" and concordance_sat > int(act)) else int(act)
		element[features.index("ACT")] = act/36 #normalize act

		####### IF CONVERTING TO NUMPY FLOAT32 THEN DELETE THIS ######START HERE########
		#convert GPA and year to number, normalize gpa
		element[features.index("GPA")] = float(element[features.index("GPA")])/5 #normalize GPA
		element[features.index("Year")] = int(element[features.index("Year")])
		####### IF CONVERTING TO NUMPY FLOAT32 THEN DELETE THIS ######END HERE########

		##this is only to show the distribition, so once you are good you can delete all instances of these lists
		gpa.append(element[features.index("GPA")])
		acts.append(element[features.index("ACT")])

		#delete SAT column because all scores have been converted to ACT
		del(element[features.index("Highest Comb SAT 1600")])

	features.remove("Highest Comb SAT 1600")
	file.close()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(gpa)
plt.subplot(1,3,2)
plt.hist(acts)
plt.subplot(1,3,3)
plt.scatter([i * 5 for i in gpa],[i * 36 for i in acts],marker='X')
plt.show()



