import sys, os
import csv, json
import pandas as pd
import re

# change its name to "df_for_{your name}.csv"
df2 = pd.read_csv("df_for_" + str(sys.argv[1]) + ".csv")

current_index = -1
if os.path.isfile('sort_comment.csv'):
    sort_comment_df = pd.read_csv('sort_comment.csv')
    current_index = len(sort_comment_df) - 1

    
for index, row in df2.iterrows():
    if index <= current_index:
        continue
        
    row["comments"] = eval(row["comments"])
    
    print("remaining post: ", len(df2) - index)
    print("post no.: ", index)
    print(row["selftext"])
    print(10 * "*")
    
    for key, value in row["comments"].items():
        print(key, "-th comment: ", value)
    
    comment_index = input("best comment: ")
    assert int(comment_index) < len(row["comments"])
    if int(comment_index) > 0:
        row["comments"]["0"], row["comments"][comment_index] = row["comments"][comment_index], row["comments"]["0"]
    
    print(20 * "=")
    print(3 * "\n")
  

    if not os.path.isfile('sort_comment.csv'):
        df2.iloc[int(index)].to_frame().T.to_csv('sort_comment.csv', index=False)
    else:
        df2.iloc[int(index)].to_frame().T.to_csv('sort_comment.csv', mode='a', header=False, index=False) 
    
