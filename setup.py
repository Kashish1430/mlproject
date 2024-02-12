from setuptools import find_packages,setup
from typing import List


def get_req(filepath:str) -> List[str]:
    '''
    Parameters
    ----------
    filepath : str
    Location to the file where all the libraries are mentioned

    Returns
    -------
    List[str]
    A list of all of the libraries that are required for the project.

    '''

    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements


setup(
name='mlproject',
version='0.0.1',
author='Kashish',
author_email='kashishrajput95@yahoo.com',
packages=find_packages(),
install_requires= get_req('requirements.txt')
)