# Llama2

[Llama2](https://ai.meta.com/llama/) is the latest collection of pretrained and fine-tuned generative text models released by Meta, ranging in scale from 7 billion to 70 billion parameters. In this example we are gonna use the Llama2-7B model to demonstrate how to get state of the art LLm models running on Lepton within just seconds.

There are two ways to access Llama2 models on Lepton:

## Fully managed Llama2 inference api

Lepton provides the standard Llama2 models as fully managed api endpoints at https://llama2.lepton.run. This api endpoint is fully compatible with OpenAI's ChatGPT API, users can directly use OpenAI's sdk or any tools that are using ChatGPT API to seamlessly switch to Llama2 model service. e.g. If you are using OpenAI's Python sdk, you can simply switch to Lepton's Llama2 inference api with

```python
import openai

openai.api_base = "https://llama2.lepton.run/api/v1"
openai.api_key = "sk-" + "a" * 48
```

After setting the `api_base` (and `api_key`) configuration, all existing code are compatible with Lepton's Llama2 inference API e.g. the following typical Python code that uses OpenAI's ChatGPT API simply works without any modifications:

```python
sys_prompt = """
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
"""
# Create a completion
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "tell me a short story"},
    ],
    stream=True,
    max_tokens=64,
)
for chunk in completion:
    content = chunk["choices"][0]["delta"].get("content")
    if content:
        print(content, end="")
print()
```

## Dedicated Llama2 inference service

If fully managed api does not fit your use case, you can also easily launch a dedicated Llama2 model inference service on Lepton platform.

Note:
Meta hosts Llama2 models weights on Huggingface. You should obtain access to these models weights by going to the corresponding model page(e.g. [llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)) and request for access. Once you have the access, go to Huggingface's [token management page](https://huggingface.co/settings/tokens) to generate a token.

### Use Lepton's secret management

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

### Launch llama2 inference service locally

Ensure that you have installed the required dependencies. Then, run:
```shell
lep photon run -n llama2 -m hf:meta-llama/Llama-2-7b-hf
```
Note that you will need to have a relatively large GPU (>20GB memory).

### Launch llama2 inference service in the cloud

Similar to other examples, you can run llama2 with the following command. Remember to pass in the huggingface access token, and also, use a reasonably sized GPU like `gpu.a10` to ensure that things run.

```shell
lep photon create -n llama2 -m hf:meta-llama/Llama-2-7b-hf
lep photon push -n llama2
lep photon run \
    -n llama2 \
    --secret HUGGING_FACE_HUB_TOKEN \
    --resource-shape gpu.a10
```

And visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n llama2 --public
```

### Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
>>> from leptonai.client import Client

>>> client = Client(...)

>>> client.run(inputs=["what is 2 + 3"], max_new_tokens=128)
"what is 2 + 3.\nThis is quite common in mathematics: variable height means variable growth and variable foot (puz- ulating, pus, pulsating), variable width for a three dimensional thing. Variable has an incorrect connotation for us. It would be better to say that the statistic is unsatisfactory in all conditions.\nBut...since he _says_ he's a 90th percentile man, and since the classification is as it is, and since those who classify him for that percentile have based it on other empirical evidence, you still have either an error in the percentile, or"
```
