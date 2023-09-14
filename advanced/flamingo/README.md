# Flamingo

[Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) is an effective and efficient general-purpose family of models that can be applied to image and video understanding tasks with minimal task-specific examples. In this example we are going to run Flamingo with [open-flamingo](https://github.com/mlfoundations/open_flamingo) on Lepton.

## Install Lepton sdk
```shell
pip install leptonai
```

## Launch Flamingo inference service locally

Run:
```shell
lep photon run -n flamingo -m photon.py
```
Although it's runnable on cpu, we recommend you to use a gpu to run vision model to get more satisfying performance.

## Launch Flamingo inference service in the cloud

Similar to other examples, you can run Flamingo with the following command.

```shell
lep photon create -n flamingo -m photon.py
lep photon push -n flamingo
lep photon run \
    -n flamingo \
    --resource-shape gpu.a10
```

Optionally, add e.g. `--env OPEN_FLAMINGO_MODEL:openflamingo/OpenFlamingo-4B-vitl-rpj3b` to specify the model you would like to run. The supported model names can be found in the open-flamingo repository's README file.

You can visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n flamingo --public
```

### Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
from leptonai.client import Client, local, current

# Use this if you are running locally
client = Client(local())
# Or, if you are logged in to your workspace via `lep login` already
# and have launched it:
# client = Client(current(), "flamingo")

inputs = {
  "demo_images": [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/test-stuff2017/000000028137.jpg"
  ],
  "demo_texts": [
    "An image of two cats.",
    "An image of a bathroom sink."
  ],
  "query_image": "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
  "query_text": "An image of"
}
res = client.run(**inputs)

print(inputs["query_text"] + res)
```

```
An image of a buffet table.
```
