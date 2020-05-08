from setuptools import setup

import experiments

setup(
        name='taras-ml-experiments',
        version=experiments.__version__,
        packages=[
                'experiments',
                'experiments.data',
                'experiments.models'
        ],
        url='www.github.com/tarasivashchuk/deep-explore',
        license='MIT',
        author='Taras Ivashchuk',
        author_email='taras@tarasivashchuk.com',
        description='Collection of experiments and personal studies in deep learning.'
)