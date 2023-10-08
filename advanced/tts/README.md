# TTS

This folder shows an end-to-end AI example, with the [Coqui AI TTS](https://github.com/coqui-ai/TTS/) text-to-speech library. The demo also shows how to run a photon with multimedia outputs (in this case a WAV response.)

With this demo, you will be able to run deepfloyd and get results like follows:

<audio src="assets/thequickbrownfox.mp3" controls></audio>

and you can check out more details in the `tts.ipynb` notebook.

## Implementation note: mounting a gradio server

In is demo, we will not only show how to create multiple paths for a photon, but also include a Gradio UI:
```python
@Photon.handler(mount=True)
def ui(self):
    blocks = gr.Blocks()
    ... # misc code that defines the blocks
    return blocks
```
The UI will then be available at the `/ui/` address. For example, if you are running locally, it would be something like `http://0.0.0.0:8080/ui/`.

## Run tts locally

Ensure that you have installed the required dependencies via `pip install -r requirements.txt`. Then, run:
```shell
python tts.py
```
Note that if you have a GPU, things will run much faster. When the program runs, visit `http://0.0.0.0:8080/ui/` for the web UI, or use the client to access it in a programmatical way.

## Run tts in the cloud

Similar to other examples, you can run tts with the following command:

```shell
lep photon create -n tts -m tts.py
lep photon push -n tts
lep photon run -n tts --resource-shape gpu.t4
```

And visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n tts --public
```

You can then use tts either via the UI or via the client. See the notebook example for more details.
