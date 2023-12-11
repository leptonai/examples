import os

from leptonai.photon import Photon


class TabbyML(Photon):
    image: str = "tabbyml/tabby"
    cmd = [
        "/opt/tabby/bin/tabby",
        "serve",
        "--model",
        os.environ.get("MODEL", "TabbyML/StarCoder-1B"),
        "--port",
        "8080",
        "--device",
        os.environ.get("DEVICE", "cuda"),
    ]

    deployment_template = {
        "resource_shape": "gpu.a10",
        "env": {
            "MODEL": "TabbyML/StarCoder-1B",
        },
        "secret": [
            "HUGGING_FACE_HUB_TOKEN",
        ],
    }
