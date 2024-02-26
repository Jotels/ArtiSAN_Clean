from distutils.util import convert_path
from typing import Dict

from setuptools import setup, find_packages

version_dict = {}  # type: Dict[str, str]

with open(convert_path('src/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='The ArtiSAN',
    version=version_dict['__version__'],
    description='',
    long_description='',
    classifiers=['Programming Language :: Python :: 3.8'],
    python_requires='>=3.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    ],
    zip_safe=False
)