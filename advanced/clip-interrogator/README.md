# clip-interrogator

[clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator) is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image. User can use the resulting prompts with text-to-image models like Stable Diffusion on DreamStudio to create cool art. In this example we are going to demonstrate how to run clip-interrogator on Lepton

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

```shell
lep photon run -n clip-interrogator -m photon.py --local
```

## Launch inference service in the cloud

Similar to other examples, you can run services on Lepton Cloud Platform easily, e.g.:

```shell
lep photon run -n clip-interrogator -m photon.py --resource-shape gpu.t4
```

You can visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n clip-interrogator --public
```

## Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
from leptonai.client import Client, local, current

# Use this if you are running locally
client = Client(local())
# Or, if you are logged in to your workspace via `lep login` already
# and have launched it:
# client = Client(current(), "clip-interrogator", token=YOUR_WORKSPACE_TOKEN)
```

```python
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
prompt = client.run(image=image)

print(prompt)
```


Image:

![two-cats](assets/two-cats.jpg)

Prompt:

```
two cats laying on a couch with remote controls on the back, on flickr in 2007, <pointé pose>;open mouth, vhs artifacts, inspired by Frédéric Bazille, long - haired siberian cat, inflateble shapes, on a hot australian day, circa 2 0 0 8, at midday, size difference, aliasing visible
```
