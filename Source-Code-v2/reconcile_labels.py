import os
import pandas as pd

def get_list_image(filename):
    return [i for i in os.listdir(filename) if i != '.DS_Store']

if __name__ == '__main__':
    ROOT_PATH = "E:\\FOR UNIVERSITY\\Special-Document\\Graduation-Project\\Images\\"
    data = pd.read_csv(os.path.join(ROOT_PATH, "Data_Entry_2017.csv"))
    samples = os.listdir(os.path.join(ROOT_PATH, "Resized-256\\"))
    samples = pd.DataFrame({"Image Index": samples})
    samples = pd.merge(samples, data, how = 'left', on = 'Image Index')
    
    samples.columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID',
                      'Patient Age', 'Patient Gender', 'View Position',
                      'OriginalImage[Width','Height]',
                      'OriginalImagePixelSpacing[x','y]', 'Unnamed']
    samples['Finding Labels'] = samples['Finding Labels'].apply(lambda x: x.split('|')[0])
    samples.drop(['OriginalImagePixelSpacing[x','y]', 'Unnamed'], axis=1, inplace=True)
    samples.drop(['OriginalImage[Width','Height]'], axis=1, inplace=True)

    print('Writing new CSV')
    samples.to_csv(os.path.join(ROOT_PATH, 'Data_samples.csv'), index=False, header=True)