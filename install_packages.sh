#!/bin/bash

# Install necessary packages
pip install -q -U wandb
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U scipy matplotlib
pip install -q plotly.express
pip install -q scikit-learn
pip install -q -U flash-attn --no-build-isolation
pip install -q pyyaml
pip install -q pyarrow
pip install -q termcolor
pip install -q pandas
pip install -q tqdm
pip install -q sqlalchemy
pip install -q python-dotenv
pip install -q aiohttp
pip install -q asyncio

echo "All packages have been installed successfully."
