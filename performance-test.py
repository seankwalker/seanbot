import json
import re
import time

MY_NAME = 'TestUser'

# Test dataset - let's make it larger for a more realistic test
test_data = [
    {"sender_name": "TestUser", "timestamp_ms": 1625097600000, "content": "Hello! How are you doing today?"},
    {"sender_name": "TestFriend", "timestamp_ms": 1625097660000, "content": "Hi there! I'm doing great, thanks for asking. How about you?"},
    # ... repeat these messages many times to create a larger dataset
] * 10000  # This will create a dataset with 20,000 messages

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('\n', ' ')
    return text.lower().strip()

def process_data_with_function():
    conversation_data = []
    for message in test_data:
        cleaned_content = clean_text(message['content'])
        if cleaned_content:
            conversation_data.append({
                'timestamp': message['timestamp_ms'],
                'sender': message['sender_name'],
                'content': cleaned_content,
                'type': 'input' if message['sender_name'] == MY_NAME else 'output'
            })
    
    data_pairs = []
    for i in range(len(conversation_data) - 1):
        if conversation_data[i]['type'] == 'input' and conversation_data[i + 1]['type'] == 'output':
            data_pairs.append((conversation_data[i]['content'], conversation_data[i + 1]['content']))
    
    return data_pairs

def process_data_inline():
    conversation_data = []
    for message in test_data:
        content = message['content']
        content = re.sub(r'http\S+', '', content)
        content = re.sub(r'[^\x00-\x7F]+', '', content)
        content = content.replace('\n', ' ')
        content = content.lower().strip()
        if content:
            conversation_data.append({
                'timestamp': message['timestamp_ms'],
                'sender': message['sender_name'],
                'content': content,
                'type': 'input' if message['sender_name'] == MY_NAME else 'output'
            })
    
    data_pairs = []
    for i in range(len(conversation_data) - 1):
        if conversation_data[i]['type'] == 'input' and conversation_data[i + 1]['type'] == 'output':
            data_pairs.append((conversation_data[i]['content'], conversation_data[i + 1]['content']))
    
    return data_pairs

# Test performance
start_time = time.time()
result_with_function = process_data_with_function()
function_time = time.time() - start_time

start_time = time.time()
result_inline = process_data_inline()
inline_time = time.time() - start_time

print(f"Time taken with function: {function_time:.4f} seconds")
print(f"Time taken with inline code: {inline_time:.4f} seconds")
print(f"Inline is {(function_time / inline_time - 1) * 100:.2f}% faster")

# Verify results are the same
print(f"Results are {'the same' if result_with_function == result_inline else 'different'}")

# Save data pairs to file (using the function version for this example)
with open('test_data.json', 'w') as f:
    json.dump(result_with_function, f, indent=2)

print("Data saved to 'test_data.json'")
