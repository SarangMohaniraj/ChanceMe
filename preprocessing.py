import csv
import numpy as np
import pandas as pd

features = None
dataset = None

college = "UMich"

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

# delete SAT column because all scores have been converted to ACT
# attend does not affect outcome
# WL and Defer are corrected for so the only decisions are accepted and denied
df.drop(["SAT","Attend","Defer","WL"], axis=1, inplace=True)

# waitlisted or deferred is accepted (it is only denied if it says denied)
#Priotity is EA and Rolling is RD
df = df.replace("Denied",0).replace(["Accepted","Waitlisted","Deferred"],1).replace("ROLL","RD").replace("PRI","EA")

# one hot encoding
df = pd.concat([df, pd.get_dummies(df["Type"])], axis=1)
# Drop the previous rank column
df = df.drop("Type", axis=1)

plot = df.copy()


# normalize
df["ACT"] = df["ACT"]/36
df["GPA"] = df["GPA"]/5
df["Year"] = (df["Year"]-df["Year"].min())/df["Year"].min()

#split dataset into training and testing
sample = np.random.choice(df.index, size=int(len(df)*0.9), replace=False)
train_data, test_data = df.iloc[sample], df.drop(sample)

features = train_data.drop('Result', axis=1)
targets = train_data['Result']
features_test = test_data.drop('Result', axis=1)
targets_test = test_data['Result']