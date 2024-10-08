## About
*Updated Oct 2024*
This repo contains training and testing code for **Feature Encapsulation for Neural Circuit Explainability (FENCE)**.

## Initial Setup
1. Acquire a server with a CUDA-compatible GPU (e.g., by renting on a cloud GPU service like Vast.ai or Runpod). You need an H100 for training, an A6000 will work fine for inference.
2. Clone this repo with `git clone https://github.com/bongohead/fence.git`.
3. Add credentials to the repo if you don't have git credentials set already.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
4. Setup a new virtual env if you need to (there is no need to for most cloud GPU services, just use the base venv). 
5. To install necessary packages, run `sh install_packages.sh`.

## Folder Structure
- The `create_phi3_from_scratch.ipynb` and `create_phi3_from_scratch_and_edit_model.ipynb` notebooks contain exploratory code where we recreate Phi3 layer-by-layer then testing the effects of editing the model structure.
- Python helpers are stored in `py_helpers`. `py_helpers.fence.forwardpass` contains code to recreate Phi3 and run forward pass with manual layer-by-layer control so that you can store/modify intermediate hidden state outputs. `py_helpers.fence.dataset` contains code to convert ChatML-formatted dictionaries into a Phi3 instruct-format string, as well as a torch dataset object which contains necessary FENCE position information.
- The notebook `train_fence_v3.ipynb` allows you to train FENCE via a notebook. Or run `python3 train-v3.py` to run it via CLI. 

## Training
- Create a `secrets.env` file with `WANDB_API_KEY` for logging.
- Run `python3 train-v3.py` to train FENCE via CLI. Models will be saved every epoch in `models/`.
- Note the training code currently uses flash attention and bfloat16 (except for self-attention modules), edit these if needed.

## Remote SSH Development
The below instructions are for setting up the correct environment on a cloud GPU server. You can disregard these if you are developing locally, though regardless you need to have access to a CUDA GPU with enough VRAM.

It's advisable to use VSCode or [Positron](https://github.com/posit-dev/positron) to run the code over SSH, instead of the default Jupyter install provided by most GPU rental providers.

Most GPU rental platforms support connecting remotely via an SSH key.
1. First, setup encryption keys on your local machine: `ssh-keygen -t ed25519`.
2. Copy the resulting key (use `cat ~/.ssh/id_ed25519.pub`) into the GPU provider's corresponding settings page (on Runpod, go to `Settings` -> `SSH Public Keys`).
3. Test whether you can connect via `ssh [username]@[ip] -p [port] -i ~/.ssh/id_ed25519`.

Next, to connect via VSCode/Positron:
1. Install the [remote-ssh](https://code.visualstudio.com/docs/remote/ssh) extension.
2. Enter the command palette and open `Remote-SSH: Open SSH Configuration File`.
3. Add the below lines to the file, substituting in the correct unix user, pot, and hostname provided by the rental service.
    ```
    <summary>
        Host gpu
        User [user]
        HostName [ip]
        IdentityFile ~/.ssh/id_ed25519
        Port [port]
    </summary>
    ```

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

Disk space can be monitered with the command `du -hs $HOME /workspace/*`.