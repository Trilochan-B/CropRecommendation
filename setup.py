from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirments(file_path:str)->List[str]:
    #this function returns requirments
    requirments = []
    with open(file_path) as f:
        requirments = f.readlines()
        requirments = [req.replace("\n","") for req in requirments]
        if HYPEN_E_DOT in requirments:
            requirments.remove(HYPEN_E_DOT)

    return requirments

setup(
    name="Project-1",
    version="0.0.1",
    author="Happy",
    author_email="trilochantrilo600@gmail.com",
    packages= find_packages(),
    install_requires=get_requirments('requirements.txt')
)