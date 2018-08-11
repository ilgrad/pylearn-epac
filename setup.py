import os
from setuptools import setup
import os.path as op

commands = [op.join('bin', 'epac_mapper')]

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="epac",
    version='0.10-1',
    author="Check contributors on https://github.com/neurospin/pylearn-epac",
    author_email="edouard.duchesnay@cea.fr",
    description=("Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow builder."),
    license="To define",
    keywords="machine learning, cross validation, permutation, parallel computing",
    url="https://github.com/neurospin/pylearn-epac",
    package_dir={'': './'},
    packages=['epac',
              'epac.map_reduce',
              'epac.sklearn_plugins',
              'epac.workflow',
              'epac.tests'],
    scripts=commands,
    long_description=read('README.md'),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.14.5',
        'joblib>=0.11',
        'scikit-learn>=0.19.0',
        'scipy>=1.0.0',
        'six>=1.11.0',
        'dill>=0.2.7.1',
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Machine learning"
    ],
)