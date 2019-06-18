from setuptools import setup, find_packages
from typing import List


def get_requirements(filename: str = 'requirements.txt') -> List[str]:
    with open(filename, 'r') as r:
        return [x.strip() for x in r.readlines()]


setup(
    name='pytorch_sound',
    version='0.0.1',
    author='ILJI CHOI',
    author_email='choiilji@gmail.com',
    description='Sound DL Package Project with Pytorch',
    keywords='sound',
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.6'
)
