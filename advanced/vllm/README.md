# vLLM

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference and serving. It seamlessly supports many Hugging Face models. In this example, we are going to show how to use vLLM to run Code Llama model on Lepton.

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
lep photon run -n codellama -m photon.py
```

By default it runs the [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) model. vLLM supports a variety of models in HuggingFace Transformers, you can use `MODEL` environment variable to let it run different model. e.g. to run [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b) model:

```shell
MODEL=tiiuae/falcon-7b lep photon run -n falcon -m photon.py
```

Note some models on HuggingFace requires executing the configuration file in that repo on your local machine, thus needs you to explicitly set `TRUST_REMOTE_CODE` environment variable to allow it, e.g. to run [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) model:

```shell
TRUST_REMOTE_CODE=ON MODEL=baichuan-inc/Baichuan2-7B-Chat lep photon run -n baichuan -m photon.py
```

Check [here](https://vllm.readthedocs.io/en/latest/models/supported_models.html) for the complete list of supported models.

## Launch inference service in the cloud

Similar to other examples, you can run services on Lepton Cloud Platform easily, e.g.:

```shell
lep photon create -n codellama -m photon.py
lep photon push -n codellama
lep photon run \
    -n codellama \
    --resource-shape gpu.a10
```

Or use `MODEL` environment variable to run a different model:

```
lep photon create -n falcon -m photon.py
lep photon push -n falcon
lep photon run \
    -n falcon \
    --env MODEL="tiiuae/falcon-7b" \
    --resource-shape gpu.a10
```

You can visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

Note: in default, the server is protected via a token, so you won't be able to access the gradio UI. This is by design to provide adequate security. If you want to make the UI public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n codellama --public
```

## Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way. The service is compatible with OpenAI sdk.

```python
import openai

SERVICE_URL = "http://localhost:8080"  # if launched locally, or get the deployment's url from dashboard on Lepton Cloud Platform
openai.api_base = f"{SERVICE_URL}/api/v1"
openai.api_key = "api-key"

completion = openai.ChatCompletion.create(
    model="codellama/CodeLlama-7b-Instruct-hf",
    messages=[
        {"role": "user", "content": "Create a user table using SQL and randomly insert 3 records"},
    ],
    stream=True,
    max_tokens=256,
)
for message in completion:
    content = message["choices"][0]["delta"].get("content")
    if content:
        print(content, end="")
print()
```

````
Sure! Here's an example of how you could create a user table in SQL and randomly insert 3 records:

```
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  email VARCHAR(100)
);

INSERT INTO users (id, name, email)
VALUES
  (1, 'John Doe', 'johndoe@example.com'),
  (2, 'Jane Doe', 'janedoe@example.com'),
  (3, 'Bob Smith', 'bobsmith@example.com');
```

This will create a table called `users` with columns for `id`, `name`, and `email`. The `id` column will be used as the primary key for the table.

The `INSERT INTO` statement is used to insert data into the table. The values for the columns are specified in the parentheses after the table name. In this case, we're inserting 3 records:

* `1, 'John Doe', 'johndoe@example.com'`
* `2, 'Jane Doe', 'janedoe@example.
````
