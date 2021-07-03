# create json file with a format catering for TFR-bert
import json

documents = []
rankingProblems = []

for index, row in df.iterrows():
  if index+1 < len(df.index) and row['sentence1'] == df.iloc[index+1]['sentence1']:
    documents.append({"relevance": row['score'], "docText": row['sentence2']})
  else:
    documents.append({"relevance": row['score'], "docText": row['sentence2']})
    rankingProblems.append({"queryText": row['sentence1'], "documents": documents})
    documents = []

df_json = {"rankingProblems": rankingProblems}

# Data to be written
with open("ltr_reddit.json", "w") as outfile:
    json.dump(df_json, outfile, ensure_ascii=False, indent=1)
