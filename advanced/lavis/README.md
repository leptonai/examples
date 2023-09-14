# LAVIS

[LAVIS](https://github.com/salesforce/LAVIS) is a Python deep learning library for LAnguage-and-VISion intelligence research and applications that supports 10+ tasks like retrieval, captioning, visual question answering (vqa), multimodal classification. In this example we are going to show how to use LAVIS to do image captioning, vqa and features extraction on Lepton.

## Install Lepton sdk
```shell
pip install leptonai
```

## Launch inference service locally

To run locally, first install dependencies:
```shell
pip install -r requirements.txt
```

After installing dependencies, you can launch inference service locally by running:

### Image Captioning

```shell
lep photon run -n caption -m caption.py
```

### Visual Question Answering (VQA)

```shell
lep photon run -n vqa -m vqa.py
```

### Features Extraction

```shell
lep photon run -n extract-features -m extract-features.py
```

## Launch inference service in the cloud

Similar to other examples, you can run services on Lepton Cloud Platform easily, e.g.:

```shell
lep photon create -n extract-features -m extract-features.py
lep photon push -n extract-features
lep photon run \
    -n extract-features \
    --resource-shape gpu.a10
```

You can visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n extract-features --public
```

## Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
from leptonai.client import Client, local, current

# Use this if you are running locally
client = Client(local())
# Or, if you are logged in to your workspace via `lep login` already
# and have launched it:
# client = Client(current(), "extract-features")  # or "caption" for Image Captioning, or "vqa" for VQA
```

### Image Captioning
```python
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
caption = client.run(image=image)

print(caption)
```

```
a couple of cats laying on top of a pink couch
```

### Visual Question Answering (VQA)

```python
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
question = "How many cats?"
answer = client.run(image=image, question=question)

print(answer)
```

```
2
```

### Features Extraction

```python
# image embedding
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
features = client.run(image=image)

print(f"embedding dimensions: {len(features)} x {len(features[0])}")
```

```
embedding dimensions: 32 x 768
```

```python
# text embedding
text = "a large fountain spewing water into the air"
features = client.run(text=text)

print(f"embedding dimensions: {len(features)} x {len(features[0])}")
```

```
embedding dimensions: 12 x 768
```

```python
# multimodal embedding
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
text = "two cats"
features = client.run(image=image, text=text)

print(f"embedding dimensions: {len(features)} x {len(features[0])}")
```

```
embedding dimensions: 32 x 768
```
