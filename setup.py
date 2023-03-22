from setuptools import setup

setup(
    name='gmphd-fusion',
    python_requires='>=3.10',
    version='0.1',
    packages=['gmphd_fusion'],
    test_suite='tests',
    url='https://github.com/mannannlegur/gmphd-fusion',
    license_files=('LICENSE',),
    author='Andrey Babushkin',
    author_email='babusand@fit.cvut.cz',
    description='GM-PHD filter for multi-target tracking with labels. Enables inclusion of '
                'additional target information in the form of external Gaussian Mixtures.'
)
