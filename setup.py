import os
import io
from glob import glob
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='PDEnsorflow',
    version='1.1.0',
    url='https://github.com/cesare-corrado/PDEnsorflow',
    author='Cesare Corrado',
    install_requires=['nibabel', 'imageio', 'numpy','scipy', 'tensorflow>=2.9'],
    author_email='cesare.corrado@kcl.ac.uk',
    description='A PDE solver using Tensorflow',
    packages=find_packages('PDEnsorflow'),
    package_dir={'': 'PDEnsorflow'},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob('PDEnsorflow/*.py')],
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        ]
)
