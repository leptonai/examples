# Stable Diffusion XL model

[Stable Diffusion XL](https://stability.ai/stablediffusion) (SDXL) is the latest open source image generation model developed by Stability AI, focusing on delivering photorealistic outputs that boast intricate details and sophisticated compositions. In this example we are demonstrate how to run an SDXL model inference service on Lepton.

There are two ways to access SDXL model:

## Fully managed SDXL inference api

Lepton provides the SDXL model as a fully managed api endpoints at https://sdxl.lepton.run. Users can easily use the lepton Python client or existing https request tool to generate high resolution realistic images right away.

Creating the client:
```python
from leptonai.client import Client

API_URL = "https://sdxl.lepton.run"
TOKEN = "YOUR_TOKEN_HERE"

c = Client(API_URL, token=TOKEN)
```

Text to Image:
```python
prompt = "A cat launching rocket"
seed = 1234
image_bytes = c.txt2img(prompt=prompt, seed=seed)
with open("txt2img_prompt.png", "wb") as f:
    f.write(image_bytes)
```

Text to Image (with refiner):
```python
prompt = "A cat launching rocket"
seed = 1234
image_bytes = c.txt2img(prompt=prompt, seed=seed, use_refiner=True)
with open("txt2img_prompt_refiner.png", "wb") as f:
    f.write(image_bytes)
```
<img src="assets/txt2img.png" width=1024>

Inpaint
```python
import base64
import requests

from leptonai.photon import FileParam


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
prompt = "A border collie sitting on a bench"
seed = 2236


# Directly using urls to pass images
image_bytes = c.inpaint(image=img_url, mask_image=mask_url, prompt=prompt, seed=seed)
with open("inpaint_url.png", "wb") as f:
    f.write(image_bytes)

# Or use FileParam to send image files:
img_content = requests.get(img_url).content
mask_content = requests.get(mask_url).content
image_bytes = c.inpaint(
    image=FileParam(img_content),
    mask_image=FileParam(mask_content),
    prompt=prompt,
    seed=seed,
)
with open("inpaint_file_param.png", "wb") as f:
    f.write(image_bytes)

# Or use base64 to encode image files:
img_content = requests.get(img_url).content
mask_content = requests.get(mask_url).content
image_bytes = c.inpaint(
    image=base64.b64encode(img_content).decode("ascii"),
    mask_image=base64.b64encode(mask_content).decode("ascii"),
    prompt=prompt,
    seed=seed,
)
with open("inpaint_base64.png", "wb") as f:
    f.write(image_bytes)
```
Image:

<img src="assets/image.png" width=512>

Mask:

<img src="assets/mask.png" width=512>

Result:

<img src="assets/inpaint.png" width=1024>

## Dedicated SDXL inference service

If fully managed api does not fit your use case, you can also easily launch a dedicated SDXL model inference service on Lepton platform.

### Launch SDXL inference service locally

Ensure that you have installed the required dependencies. Then, run:
```shell
lep photon create -n sdxl -m ./sdxl.py
lep photon run -n sdxl
```
Once the service is up, its url will be printed on the terminal screen (e.g. http://localhost:8080).

### Launch SDXL inference service in the cloud

Similar to other examples, after you have finished iterating with local service, you can launch it on Lepton cloud platform, which handles autoscaling, monitoring etc. for your production use case.

```shell
lep photon push -n sdxl
lep photon run \
    -n sdxl \
    --resource-shape gpu.a10
```

And visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to find the corresponding service url.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n sdxl --public
```

### Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
from leptonai.client import Client

SERVICE_URL = "http://localhost:8080"  # if run locally
# SERVICE_URL = "DEPLOYMENT URL shown on Lepton Cloud Platform" # if run on the Lepton Cloud Platform

c = Client(SERVICE_URL)

img_content = c.run(prompt="a cat launching rocket", seed=1234)
with open("cat.png", "wb") as fid:
    fid.write(img_content)
```
