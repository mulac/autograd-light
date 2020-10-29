import os
from setuptools import setup, find_packages

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='autograd',
      version='0.1.0',
      description='Automatic differentiation implementation for fun.  Influenced by karpathy\'s micrograd adapted for tensors.',
      author='Calum Wallbridge',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      python_requires='>=3.8',
      include_package_data=True)