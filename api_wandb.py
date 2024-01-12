
import wandb
import pandas as pd

# Replace with your W&B entity and project name
entity = "eyalb"
project="DAPT_CityScapes"

# Initialize the W&B API
api = wandb.Api(timeout=3600)

# Get all runs in the project
runs = api.runs(entity + "/" + project)

# Create an empty list to store data from each run
all_data = []

# create pd empty df:
depth_iou = pd.DataFrame()
orig_iou = pd.DataFrame()
rgb_iou = pd.DataFrame()

# Iterate through each run and extract metrics and config
# import tqdm for progress bar:
from tqdm import tqdm
# add tqdm for progress bar:

for run in tqdm(runs):
   run_hist = run.history()
   run_name = run.name

   # check if val/depth_iou is a name of colmun in the run_hist dataframe:
   if "val/depth_iou" not in run_hist.columns:
      continue

   # Append the current name as the header in new colmun in all 3 df
   depth_iou[run_name] = run_hist["val/depth_iou"]
   orig_iou[run_name] = run_hist["val/orig_iou"]
   rgb_iou[run_name] = run_hist["val/rgb_iou"]

# Save the 3 DataFrames to a CSV files:
depth_iou.to_csv("depth_iou.csv", index=False)
orig_iou.to_csv("orig_iou.csv", index=False)
rgb_iou.to_csv("rgb_iou.csv", index=False)
v=0