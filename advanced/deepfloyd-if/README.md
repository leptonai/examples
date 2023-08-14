# Deepfloyd If

This folder shows an end-to-end AI example, with the [Deepfloyd IF model](https://github.com/deep-floyd/IF). The demo also shows how to run a photon with environmental variables (or secrets) to pass in necessary credentials.

With this demo, you will be able to run deepfloyd and get results like follows:

<img src="assets/if_result.png" width=800>

## Obtain access to the deepfloyd model.
Deepfloyd hosts models on huggingface. You should obtain access as follows:
- On the [model info page](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), sign in and agree with the agreement
- Visit [tokens](https://huggingface.co/settings/tokens) page to generate the token.

## Use Lepton's secret management

As you may use the token multiple times, we recommend storing it in Lepton's secret store. Simply do this and remember to replace the token with your own.
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

## Implementation note: mounting a gradio server

In the deepfloyd example, we will not only expose a standard API, but also incorporate a UI implemented by gradio. This is done easily via the `mount` capability as follows:
```python
    @Photon.handler(mount=True)
    def ui(self) -> gr.Blocks:
        blocks = gr.Blocks()
        # Actual blocks creation code here - see deepfloyd_if.py for details.
        ...
        return blocks
```
The UI will then be available at the `/ui/` address. For example, if you are running locally, it would be something like `http://0.0.0.0:8080/ui/`.

## Run deepfloyd locally

Ensure that you have installed the required dependencies. Then, run:
```shell
python deepfloyd_if.py
```
Note that you will need to have a relatively large GPU (>20GB memory). When the program runs, visit `http://0.0.0.0:8080/ui/` for the web UI, or use the client to access it in a programmatical way.

## Run deepfloyd in the cloud

Similar to other examples, you can run deepfloyd with the following command. Remember to pass in the huggingface access token, and also, use a reasonably sized GPU like `gpu.a10` to ensure that things run.

```shell
lep photon create -n deepfloyd -m deepfloyd_if.py
lep photon push -n deepfloyd
lep photon run \
    -n deepfloyd \
    --secret HUGGING_FACE_HUB_TOKEN \
    --resource-shape gpu.a10
```

And visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n deepfloyd --public
```

You can now use deepfloyd either via the UI or via the client. Enjoy!