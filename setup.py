import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


PACKAGE_NAME = 'PaDT'
VERSION = '0.0.1'

try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A project setup for PaDT.'


INSTALL_DEPENDENCIES = [
    'torch>=2.5.1',
    'transformers==4.50.0',
    'vllm==0.6.6.post1',
    'wandb>=0.19.1',
    'pillow',
    'accelerate>=1.2.1',
    'datasets>=3.2.0',
    'deepspeed==0.15.3',
    'huggingface-hub[cli]>=0.19.2,<1.0',
    'tensorboardx',
    'qwen_vl_utils',
    'torchvision',
    'babel',
    'pycocotools',
    'httpx[socks]',
    'matplotlib',
    'scikit-image'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="Yongyi Su",
    author_email="su.yongyi.syy@gmail.com",
    description="Python setup for PaDT project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=INSTALL_DEPENDENCIES,
    entry_points={
        'console_scripts': [
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
