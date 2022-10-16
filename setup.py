from setuptools import setup, find_packages, Extension
import numpy, sys
import re

# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

def get_requires():
    reqs = []
    for line in open('docs/requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

setup(
    name='mir_tools',
    version=get_property('__version__', 'linfit'),
    author='Rayssa Guimarães Silva',
    author_email='guimaraessilvarayssa@gmail.com',
    packages=["linfit"],
    install_requires=get_requires()
    )
