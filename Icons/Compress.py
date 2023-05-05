# Import modules
from zlib import compress
from base64 import b64encode
from io import StringIO

# Load image data as bytes
with open('icon.ico', mode='rb') as rf:
    data = rf.read()

# Compress to bytes data
print(b64encode(compress(data)))
