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

        # Label the data as either 'input' or 'output' based on the sender
        for message in message_data:
            if message['sender'] == MY_NAME:
                message['role'] = 'output'
            else:
                message['role'] = 'input'

        conversation_data.append(message_data)
    # XXX For testing purposes only process the first ten conversations
    # if len(conversation_data) == 10:
    #    break

print('Data extracted and cleaned')

# Format data for training with HF SFTTrainer
# CSV of rows with columns:
# 'input': user message
# 'output': bot response
data = {'input': [], 'output': []}
for conversation in conversation_data:
    if len(conversation) == 0:
        continue

    # Build up all messages from one party into one big message until the other party responds
    current_role = conversation[0]['role']
    current_message = ''
    for i, message in enumerate(conversation):
        if message['role'] == current_role:
            # Add delimiter between sub-messages
            current_message += ' ' + message['content']
        else:
            data[current_role].append(current_message)
            current_role = message['role']
            current_message = message['content']

    # If input and output are not the same length, remove the last message
    data[current_role].append(current_message)
    if (len(data['input']) > len(data['output'])):
        data['input'].pop()
    elif (len(data['output']) > len(data['input'])):
        data['output'].pop()

    ''' XXX testing
    print('Done with one conversation')
    print(data)
    input()
    '''

print('Data formatted for training')

dataset = Dataset.from_dict(data)
dataset.to_csv('data.csv')

print('Data saved to data.csv')
