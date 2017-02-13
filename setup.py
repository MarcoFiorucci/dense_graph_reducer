import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='graph_reducer',
    version='0.1',
    description='Implementation of some constructive versions of Szemeredi Regularity Lemma for graph summarization.',
    long_description=read('README.md'),
    classifiers=['Programming Language :: Python :: 3.5',
                 'Development Status :: 4 - Beta',
                 'Topic :: Scientific/Engineering :: Image Recognition',
                 'License :: OSI Approved :: Apache Software License'],
    keywords=['Szemeredi', 'Alon', 'Frieze-Kannan', 'Graph', 'Graph Theory', 'Graph Summarization'],
    url='https://github.com/MarcoFiorucci/AlonSzemerediRegularityLemma',
    author='M. Fiorucci, A. Torcinovich',
    author_email='marco.fiorucci@unive.com',
    license='Apache 2.0',
    packages=['graph_reducer'],
    install_requires=['numpy', 'scipy'],
    package_dir={'graph_reducer': 'graph_reducer'}
)
