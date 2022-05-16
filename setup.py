#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

# could add encoding='utf-8' if needed
with open(path.join(this_directory, 'cued_sf2_lab', '_version.py')) as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cued_sf2_lab',
    version=__version__,  # noqa: F821
    license='MIT',
    description='IIA Engineering SF2 Lab',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Areeg Emarah',
    maintainer='Areeg Emarah',
    maintainer_email='ae407@cam.ac.uk',
    url='https://github.com/areeg-98/IIB_project',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'ipympl'
    ],
    # {'package_name': 'folder_with_its_source'}
    package_dir={'cued_sf2_lab': 'cued_sf2_lab'},

    classifiers=[
        # 'Intended Audience :: Science/Research',
        # 'Topic :: Scientific/Engineering :: Mathematics',

        # 'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        # "Bug Tracker": "https://github.com/pygae/clifford/issues",
        "Source Code": "https://github.com/areeg-98/IIB_project",
    },

    python_requires='>=3.5',
)
