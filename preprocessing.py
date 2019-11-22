import csv
import numpy as np
import pandas as pd

dataset = None

college = "UMD"

df = pd.read_csv("data/"+college+".csv")
# remove unwanted data
df = df[ ((df["SAT"] != "-") |  (df["ACT"] != "-")) & (df["Result"] != "Incomplete") & (df["Result"] != "No decision") & (df["Result"] != "Withdrawn") & (df["Result"] != "Unknown")]
df.reset_index(drop=True,inplace=True)

# convert SAT to ACT and choose the higher score
concordance = []
with open("concordance.csv") as file:
  concordance = list(csv.reader(file))
for index, row in df.iterrows():
  concordance_sat = int(concordance[int((1600-int(row["SAT"]))/10)][1]) if row["SAT"] != "-" else None
  df.at[index,"ACT"] = concordance_sat if row["ACT"] == "-" or (concordance_sat != None and row["ACT"] != "-" and concordance_sat > int(row["ACT"])) else row["ACT"]

df["ACT"] =  df["ACT"].astype(np.float64)
df["GPA"]= df["GPA"].astype(np.float64)
df["Year"]= df["Year"].astype(np.float64)

min_year,max_year = df["Year"].min(),df["Year"].max()

df = df.replace(["Accepted","Deferred"],1).replace(["Denied","Waitlisted"],0).replace("ROLL","RD").replace("PRI","EA")

# delete SAT column because all scores have been converted to ACT
# attend does not affect outcome
# WL and Defer are corrected for so the only decisions are accepted and denied
df.drop(["SAT","Attend","Defer","WL"], axis=1, inplace=True)

# inject noise prevent overfitting
noise = []
for act in range(37):
  for gpa in range(6):
    g = df[df["GPA"].subtract(gpa).abs() == df["GPA"].subtract(gpa).abs().min()] # datapoints with the closest gpa and various act(s)
    d1 = g["ACT"].apply(lambda x: np.abs(act - x)).idxmin() # index of the minimum of the closest act from the gpa point
    
    a = df[df["ACT"].subtract(act).abs() == df["ACT"].subtract(act).abs().min()] # datapoints with the closest act and various gpa(s)
    d2 = a["GPA"].apply(lambda x: np.abs(gpa - x)).idxmin() # index of the minimum of the closest gpa from the act point

    result = 1 if df["Result"][d1]==1 and df["Result"][d2]==1 else 0

    noise.append({"Type":df["Type"].sample().to_numpy()[0],"Result":result,"GPA":gpa,"ACT":act,"Year":np.random.randint(min_year,max_year+1)})
noise = pd.DataFrame(noise)

# one hot encoding + Drop the previous type column
df = pd.concat([df, pd.get_dummies(df["Type"])], axis=1).drop("Type", axis=1)

# save a copy of the original data before graphing, new data(with noise) gets trained
plot = df.copy()

# one hot encoding + Drop the previous type column
noise = pd.concat([noise, pd.get_dummies(noise["Type"])], axis=1).drop("Type", axis=1)
df = df.append(noise,ignore_index=True)

# normalize
df["ACT"] = df["ACT"]/36
df["GPA"] = df["GPA"]/5
df["Year"] = (df["Year"]-min_year)/max_year

"""
# balance outputs
balance = []
i = 0
while len(df[ df["Result"] == 1].index) != len(df[ df["Result"] == 0].index):
  if(i > abs(len(df[ df["Result"] == 1].index) - len(df[ df["Result"] == 0].index))):
    print(i)
    break
  x = df[ df["Result"] == df["Result"].value_counts().index[-1] ].sample()
  print(x)
  df.append(x,ignore_index=True) # adds the minority based on a random value
  print(len(df.index))
  i+=1

print(len(df[ df["Result"] == 1].index),len(df[ df["Result"] == 0].index))
"""


#split dataset into training and testing
sample = np.random.choice(df.index, size=int(len(df)*0.9), replace=False)
train_data, test_data = df.iloc[sample], df.drop(sample)

inputs = train_data.drop('Result', axis=1)
outputs = train_data['Result']
inputs_test = test_data.drop('Result', axis=1)
outputs_test = test_data['Result']
