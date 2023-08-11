# Deepfloyd If

## Sign up the user agreement with deepfloyd
On the [model info page](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), sign in and agree with the agreement

## Get the auth toke
Visit [tokens](https://huggingface.co/settings/tokens) page to generate the token

## Put in the auth token
on file deepfloyd_if.py, for every `DiffusionPipeline.from_pretrained` function called, add `use_auth_token='{YOUR_TOKEN}'` as a paramater to this function.

## Create a photon and run

```shell
lepton photon create -n deepfloyd -m py:./deepfloyd_if.py:If
lepton photon run -n deepfloyd
```

Then visit `http://localhost:8080/docs` and try it out or `http://localhost:8080/ui` to try out the UI