# OpenCLIP Example

This is a simple example of how to use the [OpenCLIP](https://github.com/mlfoundations/open_clip) to generate the embeddings of text and images. OpenCLIP is an open source implementation of OpenAI's [CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training). It is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task.
.

## Install dependencies

Within this example, we will use `conda` to manage the environment. You can install `conda` by following the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

```bash
# Create a new environment
conda create -n clip python=3.10
conda activate clip

# Install leptonai, if you've done this already, you can skip this step
pip install leptonai

# Install the dependencies
pip install -r requirements.txt
```

> During close beta stage, you may install the latest packge [here](https://www.lepton.ai/docs/overview/quickstart#1-installation)


## Create photon and run locally
    
```bash
# Create a photon
lep photon create -n clip -m open-clip.py
# Run the photon locally
lep photon run -n clip --local
```

## Make a prediction

```python
from leptonai.client import Client, local
c = Client(local())

# Embed a text
c.embed_text(query='cat')

# Embed an image by url
c.embed_image(url='https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg')

```

## Run the photon remotely

```bash
lep login # logs into the lepton cloud
lep photon push -n clip # pushes the photon to the cloud
lep photon run -n clip --resource-shape gpu.t4 # run it
```

```python
from leptonai.client import Client
LEPTON_API_TOKEN = "YOUR_LEPTON_API_TOKEN"

client = Client("YOUR_WORKSPACE_ID", "clip", token=LEPTON_API_TOKEN)

# Eg. Embed a text
result = client.embed_text(
  query="string"
)

print(result)
```