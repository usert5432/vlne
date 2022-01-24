#!/usr/bin/env python

import setuptools

def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(
    name             = 'vlne',
    version          = '0.3.0-alpha',
    author           = 'Dmitrii Torbunov',
    author_email     = 'torbu001@umn.edu',
    classifiers      = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = 'Package to train neutrino energy estimators',
    install_requires = [
        'cafplot',
        'keras',
        'pandas',
        'scipy',
        'cython',
    ],
    license          = 'MIT',
    long_description = readme(),
    packages         = setuptools.find_packages(
        exclude = [ 'tests', 'tests.*' ]
    ),
    url              = 'https://github.com/usert5432/vlne',
)

