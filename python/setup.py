from setuptools import setup
from setuptools import find_packages


setup(
    packages=find_packages(exclude=['tests']),
    test_suite='tests',

    # metadata
    name="causal_demon",
    url="https://github.com/robertness/causal_demon",
    author="Robert Osazuwa Ness",
    author_email="robertness@gmail.com",

    # dependencies
    install_requires=[
        'pyro-ppl==0.2.1',
    ]
)
