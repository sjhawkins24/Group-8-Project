from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv('OPENROUTER_API_KEY')
print(f'Key loaded: {bool(key)}')
print(f'Key length: {len(key) if key else 0}')
if key:
    print(f'Key starts with: {key[:15]}...')
else:
    print('No key found!')
