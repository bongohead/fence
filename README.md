## About
*Updated Oct 2024*
This repo contains training and testing code for **Feature Encapsulation for Neural Circuit Explainability (FENCE)**.

- First, follow the instructions below in `Setting up Runpod` below to set up a working Runpod instance.
- The `create_phi3_from_scratch.ipynb` and `create_phi3_from_scratch_and_edit_model.ipynb` notebooks contain exploratory code where we recreate Phi3 layer-by-layer then testing the effects of editing the model structure.
- Python helpers are stored in `py_helpers`. `py_helpers.fence.forwardpass` contains code to recreate Phi3 and run forward pass with manual layer-by-layer control so that you can store/modify intermediate hidden state outputs. `py_helpers.fence.dataset` contains code to convert ChatML-formatted dictionaries into a Phi3 instruct-format string, as well as a torch dataset object which contains necessary FENCE position information.
- The notebook `train_fence_v3.ipynb` allows you to train FENCE via a notebook. Or run `python3 train-v3.py` to run it via CLI. 

## Training
- Create a `secrets.env` file with `WANDB_API_KEY` for logging.
- Run `python3 train-v3.py` to train FENCE via CLI. Models will be saved every epoch in `models/`.
- Note the training code currently uses flash attention and bfloat16 (except for self-attention modules), edit these if needed.

## Setup 
The below instructions are for Runpod, but you can use another host or a local server as well, provided you have access to a CUDA GPU. 

### Initial Setup
1. Rent a Runpod instance. You need an H100 for training, an A6000 will work fine for inference.
2. SSH into the server.
3. In the terminal, `cd` to your user folder. Then, clone your repo with `git clone https://github.com/bongohead/fence.git`.
4. Add credentials to this git repo.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
5. Now after changes you should be able to push normally, `git push origin master`.

**Important:** Push VERY REGULARLY to the remote Git server. Make work as state-independent as possible, as everything not saved in Git will be destroyed every time the cloud server is turned off.

### Installing packages
Setup a new virtual env if you need to (there is no need in Runpod, just use the base venv). To install necessary packages, run `sh install_packages.sh`.

For Runpod, you should also run `sh runpod_setup.sh` to update necessary dependencies.

### Monitoring
You can use the following function to monitor GPU memory:
```
import torch

def check_memory():
    print("Allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("Reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("Total: %fGB"%(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))

check_memory()
```

Disk space can be monitered with the command `du -hs $HOME /workspace/*`. We have 100GB available in total.