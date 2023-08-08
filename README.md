<img src="assets/logo.svg" height=100>

# Lepton Examples

This repo contains a collection of sample applications you can build and run with Lepton.
Feel free to modify and use these examples as a starting point for your own applications.

The example repo is organized into the following directories:
- [getting-started](https://github.com/leptonai/examples/tree/main/getting-started): A collection of simple examples that demonstrate the basics of building and running simple photons, which are light-weight, single-file applications.
- [advanced](https://github.com/leptonai/examples/tree/main/advanced): A collection of more complex examples that demonstrate how to build and run real-world applications, such as LLMs, image search, object segmentation, and more.
- [notebooks](https://github.com/leptonai/examples/tree/main/notebooks): A collection of Jupyter notebooks that demonstrate how to use Lepton's Python SDK.

For the full documentation, please visit [https://lepton.ai/docs](https://lepton.ai/docs).

## Running examples

To run the examples, there are usually three ways:
- Directly invoking the python code to run things locally, for example:
```bash
python getting-started/counter/counter.py
# runs on local server at port 8080 if not occupied
```
- Create a photon and then run it locally with the `lep` CLI command, for example:
```bash
lep photon create -n sam -m advanced/segment-anything/sam.py
lep photon run -n sam --local
```
- Create a photon like the one above, and run it on the cloud:
```bash
lep login # logs into the lepton cloud
lep photon push -n sam # pushes the photon to the cloud
lep photon run -n sam --resource-shape gpu.t4 # run it
```
For individual examples, refer to their source files for self-explanatory comments.

## Using clients

In all three cases, you can use the python client to access the deployment via:
```python
from leptonai.client import Client, local
c = Client(local(port=8080))
```
or
```python
from leptonai.client import Client
c = Client("myworkspaceid", "sam", token="**mytoken**")
```

For example, for the `counter` example running locally, you can interact with the photon in python:
```python
>> from leptonai.client import Client, local
>> c = Client(local(port=8080))
>> print(c.add.__doc__)
Add

Automatically inferred parameters from openapi:

Input Schema (*=required):
  x*: integer

Output Schema:
  output: integer
>> c.add(x=10)
10
>> c.add(x=2)
12
```

For more details, check out the [Quickstart](https://www.lepton.ai/docs/overview/quickstart), [Walkthrough](https://www.lepton.ai/docs/walkthrough/anatomy_of_a_photon), and the [client documentation](https://www.lepton.ai/docs/walkthrough/clients).


## Contributing

We love your feedback! If you would like to suggest example use cases, please [open an issue](https://github.com/leptonai/examples/issues/new). If you would like to contribute an example, kindly create a subfolder under `getting-started` or `advanced`, and submit a pull request.
