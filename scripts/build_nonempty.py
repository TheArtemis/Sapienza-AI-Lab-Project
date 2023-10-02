# we want to build a csv of non-empty patches from the test_nir folder
import os
import numpy as np
import pandas as pd
from PIL import Image

csv_name = 'data/testing_patches_38-cloud_nonempty.csv'
NIR_FOLDER = 'data/38-Cloud_test/test_nir'
NUM_PIXELS = 384*384
patches_df = pd.DataFrame(columns=['name'])


for item in os.listdir(NIR_FOLDER):
    patch = np.array(Image.open(os.path.join(NIR_FOLDER, item)))
    
    # we add image to dataframe if it has at least 80% of nonempty pixels
    if np.count_nonzero(patch) >= 0.8*NUM_PIXELS:
        #print('Patch added!')

        patch_name = item[4:-4] # remove 'nir_' and '.TIF'
        patches_df = pd.concat([patches_df, pd.DataFrame({'name': [patch_name]})], ignore_index=True)

if os.path.exists(csv_name):
    os.remove(csv_name)

patches_df.to_csv(csv_name, index=False)
print('Done!')