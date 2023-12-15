from leptonai.photon import Photon


class ComfyUI(Photon):
    comfyui_version = "329c571"
    cmd = [
        "bash",
        "-c",
        (
            "pip install aiohttp einops torchsde &&"
            "git clone --recursive https://github.com/comfyanonymous/ComfyUI.git && cd"
            f" ComfyUI && git checkout {comfyui_version} && python main.py --listen"
            " 0.0.0.0 --port 8080"
        ),
    ]
    deployment_template = {
        "resource_shape": "gpu.a10",
    }
