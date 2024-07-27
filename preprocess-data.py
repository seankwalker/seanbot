import json
import re

MY_NAME = 'Sean Walker'

# Load JSON data
with open('messenger-export-raw.json', 'r') as file:
    data = json.load(file)

# Extract messages
messages = data['messages']

# Extract message data
message_data = []
for message in messages:
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
