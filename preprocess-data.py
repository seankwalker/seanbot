from datasets import Dataset
import json
import os
import re

MY_NAME = 'Sean Walker'

'''
Directory containing data. Each conversation is stored in subdirectory ->
message_1.json.
'''
SEARCH_DIR = os.path.join("your_facebook_activity", "messages", "inbox")

'''
    1. Traverse through each conversation in the search directory.
    2. For each conversation,
        a. Extract all messages
        b. Clean data (remove URLs, non-ASCII characters, and newlines)
        c. Label the data as either 'input' or 'output' based on the sender
    3. Format all data as input/output pairs for the model
'''

# Traverse through each conversation in the search directory
conversation_data = []
for item in os.listdir(SEARCH_DIR):
    item_path = os.path.join(SEARCH_DIR, item)

    # Skip files e.g. .DS_Store
    if not os.path.isdir(item_path):
        continue

    with open(os.path.join(SEARCH_DIR, item, 'message_1.json'), 'r') as f:
        data = json.load(f)

        # Only consider conversations with two participants because it is
        # easier to map that to input/output pairs.
        # It's possible to map a multiparty convo as well, but it's more
        # complicated.
        # TODO: Consider mapping multiparty conversations if two party convos
        #       don't produce good results.
        if len(data['participants']) != 2:
            continue

        # Extract message data
        # Skip messages that are just 'reactions' or 'sticker'
        messages = data['messages']
        message_data = []
        for message in messages:
            if 'content' not in message:
                continue
            if 'Reacted' in message['content']:
                continue
            try:
                message_data.append({
                    'timestamp': message['timestamp_ms'],
                    'sender': message['sender_name'],
                    'content': message['content']
                })
            except KeyError:
                pass

        # Clean data
        # Remove URLs, non-ASCII characters, and newlines
        for message in message_data:
            message['content'] = re.sub(r'http\S+', '', message['content'])
            message['content'] = re.sub(r'[^\x00-\x7F]+', '', message['content'])
            message['content'] = message['content'].replace('\n', '')
            message['content'] = message['content'].lower()

        # Label the data as either 'user' or 'assistant' based on the sender
        for message in message_data:
            if message['sender'] == MY_NAME:
                message['role'] = 'assistant'
            else:
                message['role'] = 'user'

        conversation_data.append(message_data)
    # XXX For testing purposes only process the first ten conversations
    # if len(conversation_data) == 10:
    #    break

print('Data extracted and cleaned')

# Format data for training with HF SFTTrainer
# i.e. { 'messages': [{ 'role': 'user', 'content': 'Hello!' }, { 'role': 'assistant', 'content': 'Hi!' }] }
# XXX: Could not get that to work, so instead just make a 2 column dataset,
#      where the first column is the role and the second column is the content
data = {'role': [], 'content': []}
for conversation in conversation_data:
    for message in conversation:
        data['role'].append(message['role'])
        data['content'].append(message['content'])

dataset = Dataset.from_dict(data)
dataset.to_csv('data.csv')
