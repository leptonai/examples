# Whisper 

This example demonstrates how to run optimized Whisper model on Lepton, and also how to integrate it with other application frameworks, in this case Slack.

[whisper-jax](https://github.com/sanchit-gandhi/whisper-jax.git) is a JAX (optimized) port of the openai whisper model. It chunks audio data into segments and then performs batch inference to gain speedup.


## Note on custom environment

Usually, when you run different AI models, they require specific dependencies that sometimes conflict with each other. This is particularly true in the whisper case - from `requirements.txt`, you may notice that there are quite a bit of specific version requirements.

This is where having a separate service like Lepton becomes super useful: we can create a python environment (using e.g. conda or virtualenv), installed the required dependencies, run the photon as a web service, and then in the regular python environment, simply call the web service as if we were using a regular python function. Comparing to some apparent choices:
- unlike a single python environment, we don't need to resolve version conflicts of different algorithms;
- unlike packing everything in a separate opaque container image, we are much more lightweighted: only a python environment and dependencies are needed.

Here we provide a combination of jax + jaxlib + cuda/cudnn pip versions that can work together inside Lepton's default image, so you can pull it up with ease.

## Running with a custom environment.

We recommend you use conda or virtualenv to start a whisper-specific environment. For example, if you use conda, it's easy to do:

```shell
# pick a python version of your favorite
conda create -n whisper python=3.10 
conda activate whisper
```

After that, install lepton [per the installation instruction](https://www.lepton.ai/docs/overview/quickstart#1-installation), and install the required dependencies of this demo via:
```shell
pip install -r requirements.txt
```

Note that `torch` sometimes has a dependency conflict with the current jax gpu libraries. If you installed torch after the above pip install and encountered errors, re-run the above installation instruction.

After this, you can launch whisper like:
```shell
# Set the model id, or omit it to use the default large-v2 model.
# Note that the large-v2 model may need ~10GB GPU memory to run.
export WHISPER_MODEL_ID="openai/whisper-medium"
python whisper-jax.py
```

It will download the paramaters and start the server. After that, use the regular python client to access the model:
```python
from leptonai.client import Client, local
c = Client(local())
```

and invoke transcription or translation as follows:
```python
>> c.run(inputs="assets/test_japanese.wav")
'私たちはAIの株式会社であります'
>> c.run(inputs="assets/test_japanese.wav", task="translate")
'We are an AI company.'
```


WE DO NOTE that for the Whisper demo, the first call will be very slow. This is because jax needs to do a bit of compilation and initialization - after that, the subsequent calls will be much faster. You may find it surprising - but for many AI deployments, the first run is usually slower due to such initialization overheads. As a good practice, if your model has such overheads, you can always do a "warm-up" call before the actual inference traffic.

## Running a slack translation bot

The whisper-jax example also demonstrates how to use Slack bot to trigger inference. To use this feature, you need to create a slack app, and set the following environment variables:
- `SLACK_VERIFICATION_TOKEN`: The verification token of your Slack app
- `SLACK_BOT_TOKEN`: The bot token of your Slack app

Let's go through the process one by one.

### Creating a slack app

First you will need to create a slack app. Go to [https://api.slack.com/apps](https://api.slack.com/apps), and click "Create an App". Choose "From scratch", and select the app name and workspace you want to add the app to, like:

<img src="assets/create_slack_app.png" width=400>

In "OAuth & Permisions", add the following permissions to the app:
- app_methions:read
- chat:write
- files:read

Which looks like the follows:

<img src="assets/permissions.png" width=400>

And install it to your workspace. After that, you should be able to see the Bot User OAuth Token like the following:

<img src="assets/bot_token.png" width=400>

Locate the verification token in the app's "Basic Information"-"App Credentials" section. Then, launch the service as follows:
```shell
export SLACK_BOT_TOKEN="xoxb-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export SLACK_VERIFICATION_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxx"
python whisper-jax.py
```

You will need a public address for slack to connect. For example, if the whisper-jax bot tells you `Uvicorn running on http://0.0.0.0:8080`, and your public IP is `1.2.3.4`, you can go to the "Event Subscriptions" page of the slack app, and enable events like the following (make sure that the request URL says "Verified"):

<img src="assets/event_subscription.png" width=400>

You should also be able to see the verification request from the logging info like
```
2023-08-09 16:35:16,119 - INFO:     34.203.249.192:47706 - "POST /slack HTTP/1.1" 200 OK
```

Now, in the "Event Subscriptions"-"Subscribe to bot events" page, add the bot user event "file_shared", and save. If everything goes well, you should be able to see the bot responding to voice inputs as follows:

<img src="assets/whisper_result.jpg" width=400>

The whisper model is fairly versatile - in this case we are showing the medium sized model, and it is already able to recognize multiple languages with a fairly good accuracy.

## Running with Lepton

The above example runs on the local machine. If your machine does not have a public facing IP, or more commonly, you want a stable server environment to host your model - then running on the Lepton cloud platform is the best option. To run it on Lepton, you can simply create a photon and push it to the cloud:

```shell
lep login
lep photon create -n whisper -m whisper-jax.py
lep photon push -n whisper
# An A10 machine is usually big enough to run the large-v2 model.
lep photon run -n whisper
  --resource-shape gpu.a10 \
  --public \
  --env WHISPER_MODEL_ID="openai/whisper-large-v2" \
  --env SLACK_BOT_TOKEN="xoxb-xxxxxxxxxxxxxxxxxxxxxxxxxx" \
  --env SLACK_VERIFICATION_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Note that because we want Slack to be able to call the API, we made it a public deployment. After that, you can use the `lep deployment status` to obtain the public address of the photon, and use the same slack app to connect to it:
```shell
>> lep deployment status -n whisper
Created at: 2023-08-09 20:24:48
Photon ID:  whisper-6t01ptsf
State:      Running
Endpoint:   https://latest-whisper.cloud.lepton.ai
Is Public:  Yes
Replicas List:
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ replica id              ┃ status ┃ message ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ whisper-f9dbd6757-l8hms │ Ready  │ (empty) │
└─────────────────────────┴────────┴─────────┘
```
When you obtain the endpoint URL above, simply change the event subscription URL to the new endpoint, such as in this case, `https://latest-whisper.cloud.lepton.ai/slack`, and you should be able to use the same slack app.

Unlike local deployment, running on the Lepton cloud platform comes with a series of advantages, especially in the whisper case:
- You do not need to worry about reproducible software environment. The photon is guaranteed to run on the same environment as you created it.
- Scaling is easier - you can simply increase the number of replicas if you need more capacity.
- Automatic fault tolerance - if the photon crashes, it will be automatically restarted.

Happy building!
