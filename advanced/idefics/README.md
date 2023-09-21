# IDEFICS

[IDEFICS](https://huggingface.co/blog/idefics) is a multimodal model that accepts sequences of images and texts as input and generates coherent text as output. It can answer questions about images, describe visual content, create stories grounded in multiple images, etc. IDEFICS is an open-access reproduction of Flamingo and is comparable in performance with the original closed-source model across various image-text understanding benchmarks. It comes in two variants - 80 billion parameters and 9 billion parameters. In this example, we are going to use the 9 billion parameters version of the model to demonstrate how to do multimodal text generation on Lepton.

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
lep photon run -n idefics -m photon.py
```

By default, the service runs [9b-instruct](HuggingFaceM4/idefics-9b-instruct) version of the model. You can use `MODEL` environment variable to select a different variant of the model to run, e.g.:

```
MODEL=HuggingFaceM4/idefics-9b lep photon run -n idefics -m photon.py
```

## Launch inference service in the cloud

Similar to other examples, you can run services on Lepton Cloud Platform easily, e.g.:

```shell
lep photon create -n idefics -m photon.py
lep photon push -n idefics
lep photon run \
    -n idefics \
    --resource-shape gpu.a10
```

By default, the service runs [9b-instruct](HuggingFaceM4/idefics-9b-instruct) version of the model. You can use `MODEL` environment variable to select a different variant of the model to run, e.g.:

```shell
lep photon create -n idefics -m photon.py
lep photon push -n idefics
lep photon run \
    -n idefics \
    --env MODEL="HuggingFaceM4/idefics-9b" \
    --resource-shape gpu.a10
```

You can visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

If you want to make the api public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n idefics --public
```

## Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
from leptonai.client import Client, local, current

# Use this if you are running locally
client = Client(local())
# Or, if you are logged in to your workspace via `lep login` already
# and have launched it:
# client = Client(current(), "idefics", stream=True)
```

```python
image = "https://huggingfacem4-idefics-playground.hf.space/file=/home/user/app/example_images/obama-harry-potter.jpg"
question = "Which famous person does the person in the image look like? Could you craft an engaging narrative featuring this character from the image as the main protagonist?"
eos_token = "<end_of_utterance>"
prompts = [
    f"User: {question}",
    image,
    eos_token,
    "\nAssistant:",
]
res = client.run(prompts=prompts)
print(res)
```

```
User: Which famous person does the person in the image look like? Could you craft an engaging narrative featuring this character from the image as the main protagonist?
Assistant: The person in the image looks like Harry Potter, the famous wizard from the Harry Potter book series. As the main protagonist, Harry Potter embarks on a thrilling adventure to defeat the evil Lord Voldemort and save the wizarding world from his grasp. Along the way, he makes new friends, learns powerful spells, and discovers the true extent of his own magical abilities. With the help of his loyal companions Hermione Granger and Ron Weasley, Harry Potter faces countless challenges and obstacles, ultimately emerging victorious and becoming a legend in the wizarding world.
```
