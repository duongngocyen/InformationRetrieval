from datasets import load_dataset
import openai
import pandas as pd
import time
from tqdm import tqdm
import json
from openai import OpenAI
key=[FILL-IN-YOUR-KEY]
client = OpenAI(api_key=key)
def get_chatgpt_answer(input_prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
        {"role": "system", "content": f"You are helpful assistant for sentiment analysis task."},
        {"role": "user", "content": input_prompt},
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def get_chatgpt_answer_with_retry(prompt, max_retries=3, retry_delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            answer = get_chatgpt_answer(prompt)
            return answer
        except Exception as e:
            print(f"Error in answer API call (attempt {attempt}/{max_retries}): {str(e)}")
            time.sleep(retry_delay * attempt)
    raise e

TEST_CASE='llm'

for job in [""]:
    data = pd.read_csv(f"../data/FINAL_TEST_DATA.csv")
    raws = []
    outputs = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        text = row.Content
        prompt = f"""
You are asked to analyse the following post from reddit and assign a rating according to the post.

### START INPUT ###
{text}
### END INPUT ###

You should answer in following format:
1. Identify the main object of the post (either IOS and Android)
Main object:...
2. Find the rating of the text toward the object. Final rating must be in ['negative', 'neutral', 'positive']. If you are not sure, output 'neutral'
Sentiment: [Choose in 'negative' OR 'neutral' OR 'positive']
    """
        raw_ans = get_chatgpt_answer_with_retry(prompt)
        try:
            ans = raw_ans.split("Sentiment:")[1].strip()
        except:
            try:
                ans = raw_ans.split("Sentiment")[1].strip()
            except:
                ans = raw_ans
        print(ans)
        raws.append(raw_ans)
        outputs.append(ans.lower())
        test = data[:index+1]
        test['raw_gpt'] = raws
        test['guide_gpt'] = outputs
        test.to_csv(f"../data/final_test_gpt.csv", index=False)