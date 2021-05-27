from urllib.request import urlopen
import sys, os
import csv, json
import pandas as pd
import praw 

# login reddit as a developer

reddit = praw.Reddit(client_id='k1tLqpeRpb9m6A',
                     client_secret='o1FuiZenF-nPGfMzW9eeFrN6M2k',
                     user_agent='praw',
                     )



def harvest_posts_comments(name):
    subreddit = reddit.subreddit(name)
    subreddit = subreddit.top(limit=None)
    
    print(name)
    for submission in subreddit:  # extrat each post of a subreddit
        DataFrame = pd.DataFrame()

        if not submission.stickied:
            post = {"subreddit": submission.subreddit.display_name,
                  "subid": submission.id,
                  "title": submission.title,
                  "selftext": submission.selftext}
            
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            comments = {}
            i = 0
            for comment in submission.comments.list()[:10]: # extrat best 10 comments of a post
                comment = {"comment_id":comment.id,
                           "upvote": comment.score,
                           "body": comment.body}
                comments[str(i)] = comment
                i += 1
            post["comments"] = comments
            
            df = pd.DataFrame.from_dict(post, orient='index').T
            if DataFrame.empty:
                DataFrame = df
            else:
                DataFrame = DataFrame.append(df, ignore_index=True)

            # write to .csv file
            if not os.path.isfile('./text.csv'):
                DataFrame.to_csv('./text.csv', index=False)
            else:
                DataFrame.to_csv('./text.csv', mode='a', header=False, index=False) 
                
            print(DataFrame)


SUB_REDDITS = ["confession", "raisingkids", "AskParents", "family", "AttachmentParenting", "Nanny"]

print("start scrappering")
for subreddit in SUB_REDDITS:
    harvest_posts_comments(subreddit)
    
print("done")
df = pd.read_csv('text.csv')
for index, row in df.iterrows():
    row.comments = eval(row.comments)
    

print("convert to json format")
result = df.to_json("text.json", orient="index")
print("done")
