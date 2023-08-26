# CodeLlama

[CodeLlama](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) released by Meta is a family of LLM based on Llama 2, providing code completion, infilling capabilities and zero-shot instruction following ability for programming tasks. In this example we are going to demonstrate how to run CodeLlama-7b model on Lepton.

At the point of writing, running CodeLlama models relies on some relatively [new changes](https://github.com/huggingface/transformers/pull/25740) in HuggingFace Transformers that are not released yet, so please make sure to install transformers from source until the next version is released:

`pip install git+https://github.com/huggingface/transformers.git@015f8e1 accelerate`

## Launch CodeLlama inference service locally

Ensure that you have installed the required dependencies. Then, run:
```shell
lep photon run -n codellama -m photon.py
```
Note that you will need to have a relatively large GPU (>=16GB memory).

Use `MODEL` environment variable to switch to a different model in the CodeLlama family, e.g.

```shell
MODEL=codellama/CodeLlama-7b-Instruct-hf lep photon run -n codellama -m photon.py
```

## Launch CodeLlama inference service in the cloud

Similar to other examples, you can run CodeLlama with the following command. Use a reasonably sized GPU like `gpu.a10` to ensure that things run.

```shell
lep photon create -n codellama -m photon.py
lep photon push -n codellama
lep photon run \
    -n codellama \
    --resource-shape gpu.a10
```

Use `MODEL` environment variable to switch to a different model in the CodeLlama family, e.g.

```shell
lep photon run \
    -n codellama \
    --env MODEL=codellama/CodeLlama-7b-Instruct-hf \
    --resource-shape gpu.a10
```

And visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n codellama --public
```

### Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

Create client
```python
>>> from leptonai.client import Client, local

>>> client = Client(local(port=8080))  # If the inference service was launched in the cloud, change the parameters to create the client, see https://github.com/leptonai/examples#using-clients
```

Code completion:
```python

>>> prompt = '''\
import socket

def ping_exponential_backoff(host: str):
'''

>>> print(client.run(inputs=prompt, max_new_tokens=256))
'''
import socket

def ping_exponential_backoff(host: str):
    """Repeatedly try until ping succeeds"""
    for i in range(1,11):
        print('Ping attempt '+str(i))
...
'''
```

If you have chosen to use the "Instruct" models (e.g. the "codellama/CodeLlama-7b-Instruct-hf" one mentioned above), you can instruct/chat with the model:

Instructions/Chat:
````python
>>> user = 'In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?'

>>> prompt = f"<s>[INST] {user.strip()} [/INST]"

>>> print(client.run(inputs=prompt, max_new_tokens=256)[len(prompt):])
'''
You can use the `find` command in Bash to list all text files in the current directory that have been modified in the last month. Here's an example command:
```
find. -type f -name "*.txt" -mtime -30
```
Here's how the command works:

* `.` is the current directory.
* `-type f` specifies that we want to find files (not directories).
* `-name "*.txt"` specifies that we want to find files with the `.txt` extension.
* `-mtime -30` specifies that we want to find files that have been modified in the last 30 days.

The `-mtime` option takes a number of days as its argument, and the `-30` argument means "modified in the last 30 days".
...
'''
````
