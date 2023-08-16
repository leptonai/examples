# Custom Image

Lepton supports the usage of custom images with your own software environment,
given the following conditions:
- the image is a relatively standard Linux image, and
- it contains `python` (>3.7) and `pip`, and
- optionally, to install system dependencies, it should support `apt`.

Note: despite the fact that custom images are very flexible, you should use
the default image if possible, and use `requirement_dependencies` and
`system_dependencies` to install dependencies. This is because in the cloud
environment, we do a lot to minimize the loading time of the default image,
and a custom image may take much longer (at the scale of minutes) to load.

To specify custom image is simple: in your Photon class, simply specify
```python
class MyPhoton(Photon):
    image="your_custom_image_location"
```

To build the example, simply do:

    lep photon create -n custom-image -m custom-image.py

To run the photon, simply do

    lep photon push -n custom-image
    lep photon run -n custom-image [optional arguments]

