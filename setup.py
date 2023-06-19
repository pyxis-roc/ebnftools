# SPDX-FileCopyrightText: 2020,2021 University of Rochester
#
# SPDX-License-Identifier: MIT

from setuptools import setup, find_packages

setup(name='ebnftools',
      version='0.1',
      packages=find_packages(),
      scripts=['ebnftools/cvt2bnf.py']
)
