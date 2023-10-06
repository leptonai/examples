# Nougat

[Nougat](https://github.com/facebookresearch/nougat) (Neural Optical Understanding for Academic Documents) is a Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language. In this example, we are going to show how to use Nougat to turn scanned PDF files (human readable documents) to markups (machine-readable text).

## Install Lepton sdk
```shell
pip install leptonai
```

## Launch inference service locally

To run locally, first install dependencies:
```shell
pip install -r requirements.txt
```

Nougat uses `pdfinfo` to extract the "Info" section from PDF files, thus need to install `poppler-utils`:

```shell
sudo apt-get update
sudo apt-get install poppler-utils
```

After installing dependencies, you can launch inference service locally by running:

```shell
lep photon run -n nougat -m photon.py
```

## Launch inference service in the cloud

Similar to other examples, you can run services on Lepton Cloud Platform easily, e.g.:

```shell
lep photon create -n nougat -m photon.py
lep photon push -n nougat
lep photon run \
    -n nougat \
    --resource-shape gpu.a10
```

You can visit [dashboard.lepton.ai](https://dashboard.lepton.ai/) to try out the model.

If you want to make the api public, you can either add the `--public` argument to `lep photon run`, or update the deployment with:

```shell
lep deployment update -n nougat --public
```

## Client

Once the inference service is up (either locally or in the cloud), you can use the client to access it in a programmatical way:

```python
from leptonai.client import Client, local, current

# Use this if you are running locally
client = Client(local(), stream=True)
# Or, if you are logged in to your workspace via `lep login` already
# and have launched it:
# client = Client(current(), "nougat", stream=True)
```

```python
PDF_FILE = "https://www.gcpsk12.org/site/handlers/filedownload.ashx?moduleinstanceid=74914&dataid=140852&FileName=Sample%20Scanned%20PDF.pdf"
content_iter = client.run(file=PDF_FILE)
for chunk in content_iter:
    print(chunk.decode("utf-8"))
```

```
Document Title (Heading Style 1)

Topic 1 (Heading Style 2)

Normal Paragraph Style: Lorentz ipsum dolor sit amet, consecetetur adipiscing elit, sed do

elusmod temper incididunt ut labore et dolore magna aliquua. Dapibus uttrices in iaculis

nunc sed augue. Fusce ut placerat orci nulla pellentesque dignissim enim sit. Nunc

congue nisi vitae suscipitt tellus. Tristique et egestas quis ipsum suspendisse uttrices.

Nunc aliquet bibendum enim facilis gravida neque.

Topic 2 (Heading Style 2)

Subtopic A (Heading Style 3)
...
```
