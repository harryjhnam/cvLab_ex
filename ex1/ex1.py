import os
import sys

path = '/Users/HarryJHNam/Environments/python_ex/'
files = os.listdir(path)

arg = sys.argv[1]
"""
if arg=="name":
    print("sorting by name: ", sorted(files, key=str.lower))
elif arg=="date":
    print("sorting by date: ", sorted(files, key=os.path.getctime))
else :
    print("files: ",files)
"""

def file_sort(sort_type):
    return {
        'name': sorted(files, key=str.lower),
        'date': sorted(files, key=os.path.getctime),
        }[arg]

print("sorting by ", arg, ":", file_sort(arg))
