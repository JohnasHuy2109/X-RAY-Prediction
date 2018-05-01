import h2o
import os
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator


def drop_column(df, lst):
    df.drop(lst, axis=1, inplace=True)
    return df

def get_training_columns(df, target):
    return [col for col in df.columns if col != target]

if __name__ == "__main__":
    ROOT_PATH = "E:\\FOR UNIVERSITY\\Special-Document\\Graduation-Project\\Images\\"
    
    data = pd.read_csv(os.path.join(ROOT_PATH, 'Data_Entry_2017.csv'), skiprows=1, names = ['Image Index', 'Finding Labels', 'Follow-up #',
                                                                         'Patient ID', 'Patient Age', 'Patient Gender',
                                                                         'View Position', 'OriginalImage[Width',
                                                                         'Height]',
                                                                         'OriginalImagePixelSpacing[x',
                                                                         'y]',
                                                                         'Unnamed'], low_memory=False)
    data = drop_column(data, ['Follow-up #', 'Unnamed', 'OriginalImage[Width','Height]',
                       'Image Index', 'Patient ID', 'OriginalImagePixelSpacing[x','y]'])
    
    data['Patient Age'] = data['Patient Age'].map(int, lambda x: str(x)[:-1]).astype(int)
    data['Finding Labels'] = data['Finding Labels'].apply(lambda x: x.split('|')[0])
    
    h2o.init()
    
    data = h2o.H2OFrame(data)
    
    train, valid, test = data.split_frame(ratios=[0.6, 0.2], seed=8)
    
    training_columns = get_training_columns(train, 'Finding Labels')
    
    gbm = H2OGradientBoostingEstimator(ntrees=1000, distribution='multinomial', max_depth=2, learn_rate=0.001, balance_classes=True)
    gbm.train(x=training_columns, y='Finding Labels', training_frame=train, validation_frame=valid)