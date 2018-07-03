
import sys
import matplotlib.pyplot as plt

import func

text = open('lorem_ipsum.txt','r+').read()

arg = sys.argv

arr = func.split_text(text, arg)

plt.hist(arr)
plt.show()

