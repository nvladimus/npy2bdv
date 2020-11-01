import os
import shutil
import glob
print("Building docs using pdoc module")
os.system("pdoc npy2bdv --html -o ./docs")
files = glob.glob("./docs/npy2bdv/*.html")
for f in files:
    shutil.copy(f, "./docs")
shutil.rmtree("./docs/npy2bdv")
os.listdir("./docs")