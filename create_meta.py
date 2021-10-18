import os

import pandas as pd

final_test_dir = 'dataset/classification/val'
final_test_imgs = os.listdir(final_test_dir)
final_test_imgs =[os.path.join(final_test_dir, img) for img in final_test_imgs]
classes = [-1] * len(final_test_imgs)

df =pd.DataFrame()
df['img_path'] = pd.DataFrame(final_test_imgs)
df['label'] = pd.DataFrame(classes)
df.to_csv('dataset/classification/val.csv',index=False)