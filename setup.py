#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ROOT_PACKAGE_NAME = 'src'


def parse_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name=ROOT_PACKAGE_NAME,
    version='1.0',
    author=['Shlychkov Konstantin'],
    packages=find_packages(),
    long_description='feature extractor',
    requirements=parse_requirements()
)

