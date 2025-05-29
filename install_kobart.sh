#!/bin/bash

# Install KoBART
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart

# Install TPU-specific dependencies
pip install torch==2.7.0
pip install torch_xla==2.7.0