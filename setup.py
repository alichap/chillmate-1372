from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='base_fruit_classifier',
    version="0.0.1",
    description="basic fruit classifier for chillmate project",
    author="Chillmate",
    author_email="",
    install_requires=requirements,
    packages=["base_fruit_classifier"],

      )
