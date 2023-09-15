<img src="assets/logo.svg" height=100>

# Lepton Examples

This repository contains a collection of example photons built with the [Lepton AI sdk](https://github.com/leptonai/leptonai-sdk/).

Feel free to modify and use these examples as a starting point for your own applications.

The example repo is organized into the following directories:
- [getting-started](https://github.com/leptonai/examples/tree/main/getting-started): A collection of simple examples that demonstrate the basics of building and running simple photons, which are light-weight, single-file applications.
- [advanced](https://github.com/leptonai/examples/tree/main/advanced): A collection of more complex examples that demonstrate how to build and run real-world applications, such as LLMs, image search, object segmentation, and more.

For the full documentation, please visit [https://lepton.ai/docs](https://lepton.ai/docs).

## Prerequisite

Note: we are currently in closed beta. All examples in this folder are runnable locally. To run on the Lepton cloud, join the waiting list at [https://lepton.ai/](https://lepton.ai/).

Install the lepton sdk via (the `-U` option ensures the most recent version is installed):
```python
pip install -U leptonai
```

For many examples in the [advanced](https://github.com/leptonai/examples/tree/main/advanced) folder, there are dependencies needed by the specific algorithms. It is recommended that you use virtual environments to not pollute your daily environment. For example, if you use conda, you can do:
```shell
conda create -n myenvironment python=3.10
conda activate myenvironment
```

## Running examples

To run the examples in the respective folders, there are usually three ways:
- Directly invoking the python code to run things locally, for example:
```bash
python getting-started/counter/counter.py
# runs on local server at port 8080 if not occupied
```
- Create a photon and then run it locally with the `lep` CLI command, for example:
```bash
lep photon create -n sam -m advanced/segment-anything/sam.py
lep photon run -n sam --local
```
- Create a photon like the one above, and run it on the cloud:
```bash
lep login # logs into the lepton cloud
lep photon push -n sam # pushes the photon to the cloud
lep photon run -n sam --resource-shape gpu.t4 # run it
```
For individual examples, refer to their source files for self-explanatory comments.

## Using clients

In all three cases, you can use the python client to access the deployment via:
```python
from leptonai.client import Client, local
c = Client(local(port=8080))
```
or
```python
from leptonai.client import Client
c = Client("myworkspaceid", "sam", token="**mytoken**")
```

For example, for the `counter` example running locally, you can interact with the photon in python:
```python
>> from leptonai.client import Client, local
>> c = Client(local(port=8080))
>> print(c.add.__doc__)
Add

Automatically inferred parameters from openapi:

Input Schema (*=required):
  x*: integer

Output Schema:
  output: integer
>> c.add(x=10)
10
>> c.add(x=2)
12
```

For more details, check out the [Quickstart](https://www.lepton.ai/docs/overview/quickstart), [Walkthrough](https://www.lepton.ai/docs/walkthrough/anatomy_of_a_photon), and the [client documentation](https://www.lepton.ai/docs/walkthrough/clients).


## Notes on huggingface access

Sometimes, you might encounter errors accessing huggingface models, such as the following message when accessing `llama2`:
```text
Failed to create photon: 401 Client Error. (Request ID: Root=xxxxxxx)

Cannot access gated repo for url https://huggingface.co/api/models/meta-llama/Llama-2-7b-hf.
Repo model meta-llama/Llama-2-7b-hf is gated. You must be authenticated to access it.
```
This means that you did not have access to the repo, or you did not set up huggingface access tokens. We'll detail how to do so below.

### Get access to the huggingface repo.
Go to the corresponding huggingface repo, and accept the terms and conditions of the corresponding repository. For example, for llama2, the corresponding link is [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).

### Set up credentials to access huggingface
Your credential to access huggingface can be found online at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

If you are running photons locally, you can do one of the following:
- set the token as an environmental variable, with `export HUGGING_FACE_HUB_TOKEN=your_token_here`.
- or, in your python environment, run the following command and login. Huggingface will store the credential in the local cache, usually `.huggingface/token`, for repeated usage:
```python
import huggingface_hub
huggingface_hub.login()
```

If you are running on the Lepton cloud remotely, the easiest approach is to use the `secret` feature of Lepton. You can safely store the huggingface token as a secret via CLI:
```shell
lep secret create -n HUGGING_FACE_HUB_TOKEN -v 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
```
(Don't worry, the above token is only an example and isn't active.)

You can verify the secret exists with `lep secret list`:
```shell
>> lep secret list
               Secrets               
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ ID                     ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ HUGGING_FACE_HUB_TOKEN │ (hidden) │
└────────────────────────┴──────────┘
```

And when you launch a photon, add `--secret`:
```shell
lep photon run -n myphoton --secret HUGGING_FACE_HUB_TOKEN
```

For more detailed information, check out the following resources:
- [Huggingface's login reference](https://huggingface.co/docs/huggingface_hub/package_reference/login)
- [Lepton documentation on secrets](https://www.lepton.ai/docs/advanced/env_n_secrets)
- [An example showing huggingface access using the deepfloyd-if model](https://github.com/leptonai/examples/tree/main/advanced/deepfloyd-if)

## Contributing

We love your feedback! If you would like to suggest example use cases, please [open an issue](https://github.com/leptonai/examples/issues/new). If you would like to contribute an example, kindly create a subfolder under `getting-started` or `advanced`, and submit a pull request.
