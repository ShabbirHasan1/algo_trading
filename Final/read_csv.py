import pandas as pd
import glob
import os

path = r'C:\Desktop\Learning\Courses\zerodha\exp' # use your path

option = glob.glob(path + "/V2_delayed*.csv")

data = {}

for i, file in enumerate(option):
    data[file] = pd.read_csv(file)
    # data[file].columns = list(data[file].columns[:18]) + ['t1','t2','t3','t4']
    data[file]['Name'] = os.path.basename(file)[-9:-4]
    data[file]['Name']
df = pd.concat([data[file] for file in option],axis=0)

os.chdir("C:/Desktop/Learning/Courses/zerodha/exp")
df.to_csv("V1_delayed_final.csv",index = False)   


