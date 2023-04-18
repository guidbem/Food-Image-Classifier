from utils import *
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split


# run the Python script "unzip_image_folder.py"
subprocess.run(['python', 'unzip_image_folder.py'])

# Import the train_val and test metadata
if os.path.exists('train_val.csv') and os.path.exists('test.csv'):
    train_val_df = pd.read_csv('train_val.csv')
    test_df = pd.read_csv('test.csv')
else:
    df_imgs = prepare_datafile('image_metadata.csv', 'data')
    train_val_df, test_df = train_test_split(df_imgs, train_size=0.8)
    train_val_df.to_csv('train_val.csv', index=False)
    test_df.to_csv('test.csv', index=False)


train_df, val_df = train_test_split(train_val_df, train_size=0.875)

model = ImageClassifier(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    params={
        'optimizer':'Adam',
        'lr':0.001
    }
)

best_model, cv_val_loss, cv_val_auc = cross_val_loop(
    train_val_df=train_val_df,
    test_df=test_df,
    tuning_params={
        'optimizer': 'Adam',
        'lr': 0.001
    }
)