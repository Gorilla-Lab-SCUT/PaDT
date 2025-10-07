#!/bin/bash
# conda create -n PaDT python=3.11
# conda activate PaDT

pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
