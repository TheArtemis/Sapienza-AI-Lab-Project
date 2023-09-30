import os
import numpy as np
import pandas as pd
from PIL import Image

#check if path '../data/38-Cloud_test/Entire_scene_gts' exists
if not os.path.exists('data/38-Cloud_test/Entire_scene_gts'):
    raise Exception('Path not found: ../data/38-Cloud_test/Entire_scene_gts\n Please run 38-cloud-notebook.ipynb first')

#make dataframe from 'data/38-Cloud_test/test_sceneids_38-Cloud.csv
df = pd.read_csv('data/38-Cloud_test/test_sceneids_38-Cloud.csv')
img_names = df['Landsat 8 Collection 1 Level 1 Product SceneID'].rename('name')
df = None

if not os.path.exists('data/38-Cloud_test/test_gt'):
    os.mkdir('data/38-Cloud_test/test_gt')

# Landsat 8 Collection 1 Level 1 Product SceneID

#for each image in img_names we want to add margins so it can be divided into 384x384 patches
for name in img_names:
    print(f'Processing: {name}')
    img = np.array(Image.open('data/38-Cloud_test/Entire_scene_gts/edited_corrected_gts_' + name + '.TIF'))
    img_x, img_y = img.shape

    #print(img)
    
    margin_x = 0
    margin_y = 0

    if img_x % 384 != 0:
        margin_x = 384 - img_x % 384
    if img_y % 384 != 0:
        margin_y = 384 - img_y % 384    

    new_img = np.zeros((img_x + margin_x, img_y + margin_y))
    new_img = np.pad(img, ((margin_x//2, margin_x - margin_x//2), (margin_y//2, margin_y - margin_y//2)), mode='constant')
    new_img = np.where(new_img == True, 255, 0).astype(np.uint8)     
    s = Image.fromarray(new_img) # TEMPORARY
    s.save('test/' + name + '.TIF') # TEMPORARY

    x = 1
    for i in range(0, new_img.shape[0], 384):
        for j in range(0, new_img.shape[1], 384):
            patch = new_img[i:i+384, j:j+384]                  
            #print(patch)
            patch = Image.fromarray(patch)            
            patch_name = f'data/38-Cloud_test/test_gt/gt_patch_{x}_{(i//384+1)}_by_{(j//384+1)}_{name}.TIF'            
            #print('saving: ', f'data/38-Cloud_test/test-gt/gt_patch_{i//384+j//384}_{i//384}_{j//384}_{name}.TIF')
            patch.save(patch_name)
            x+=1

print('Done!')

