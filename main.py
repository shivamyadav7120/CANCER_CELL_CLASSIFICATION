import os

python_path = r"C:\Users\Lenovo\anaconda3\envs\cancer\python.exe"

print("Step 1: Training Model")
os.system(f'"{python_path}" train.py')

print("Step 2: Predicting")
os.system(f'"{python_path}" predict.py')

print("Completed Successfully!")