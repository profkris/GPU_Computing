## Steps to Compile and Run CUDA Codes on Google Colab or Kaggle

### Step 1: Open an Open-Source Platform

Open either Google Colab or Kaggle in your browser.
----------------------------------------------------------------------------------------------------
### Step 2: Check GPU Availability

In a code cell, run the following command to check if a GPU is available:

!nvidia-smi

/bin/bash: nvidia-smi: command not found

If nvidia-smi is not found, proceed to Step 3.
-----------------------------------------------------------------------------------------------------
## Step 3: Select Runtime as GPU from settings

again check if a GPU is available:
you will get this kind of massesge if GPU is set properly

| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   37C    P8              9W /   70W |       1MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
| N/A   34C    P8              9W /   70W |       1MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
-------------------------------------------------------------------------------------------------------------------
## Step 4: In same code cell write code along with %%writefile <code_name.cu>

eg. 
%%writefile test.cu
// Your CUDA code here

---------------------------------------------------------------------------------------------------------------------

## Step 5: Compile the CUDA code using nvcc:

! nvcc <code_name.cu> -o <any_executable_name>
eg. ! nvcc test.cu -o test

----------------------------------------------------------------------------------------------------------------------

## Step 6: Execute your code

! nvcc ./<executable_name>
eg. !./test


