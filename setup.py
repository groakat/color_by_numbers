from __future__ import with_statement
import os
from setuptools import find_packages

from distutils.core import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "colorbynumbers",
    version = "0.1.0",
    author = "Peter Rennert",
    author_email = "github@rennert.io",
    description = ("A simple script to convert an RGB image into a color by numbers image"),
    packages=find_packages(),
    #license = read('LICENSE.txt'),
    keywords = "audio",
    url = "https://github.com/groakat/colorbynumbers",
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
    ],
)
