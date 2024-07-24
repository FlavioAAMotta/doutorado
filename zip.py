from zipfile import ZipFile
import os

# the folder is 'results' which is in the same directory as this script
folder = os.path.dirname(os.path.abspath(__file__)) + "/results/"

# zip the files recursively
def zip_files():
    with ZipFile('results.zip', 'w') as zip:
        for root, dirs, files in os.walk(folder):
            for file in files:
                zip.write(os.path.join(root, file))
                    
zip_files()