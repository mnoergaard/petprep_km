from setuptools import setup, find_packages

setup(
    name='petprep_km',
    version='0.1',
    packages=find_packages(include=['petprep_km', 'petprep_km.*']),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'statsmodels'
    ],
    entry_points={
        'console_scripts': [
            'petprep_km=petprep_km.cli.run:main',
        ],
    },
)
