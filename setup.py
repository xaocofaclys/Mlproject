from setuptools import setup, find_packages
from pathlib import Path
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements from the given file path.
    :param file_path: str: Path to the requirements file
    :return: list: List of requirements
    """
    with open(file_path) as file:
        lines = file.readlines()
        # Remove any comments and strip whitespace
        abc =  [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        if '-e .' in abc:
            abc.remove('-e .')

setup(
    name='mlproject',
    version='0.1',
    author='shubham',
    author_email = 'xaocofachlys@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A simple ML project',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    )