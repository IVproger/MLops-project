'''
Import the sample_data function from the data.py script and test it.
'''
from omegaconf import OmegaConf
from src.data import sample_data

def test1_sample_data():
    '''
    Check that the data.py script is working correctly and returning the sample data.
    '''
    cfg = OmegaConf.load("../configs/main.yaml")
    sample = sample_data(cfg)
    assert sample is not None
   
def test2_sample_data():
    '''
    Check the sample_data function from the data.py script when the output path is not defined.
    '''
    cfg = {
        'data': {
            'url': 'https://drive.google.com/uc?id=1CKXjF6xZEgjYsdwfCMxoHChI7J5WjjdI',
            'output': '',
            'datastore_path': '../datastore/data.csv',
            'sample_path': '../data/samples/sample.csv',
            'sample_size': 0.2
        }
    }
    sample = sample_data(cfg)
    assert sample is None
    