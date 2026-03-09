# save uploaded files content in temp file
import os

def save_uploaded(content, file_path):
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False