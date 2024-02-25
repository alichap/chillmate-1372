from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='base_fruit_classifier',
      version="0.0.1",
      description="basic fruit classifier for chillmate project",
<<<<<<< HEAD
      author="Chillmate",
=======
      author="Chillmates",
>>>>>>> main
      authors_email="",
      #url="https://github.com/lewagon/taxi-fare",
      install_requires=requirements,
      packages=find_packages(),
      #packages=["base_fruit_classifier"],
      #test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      #include_package_data=True,
      #zip_safe=False
      )
