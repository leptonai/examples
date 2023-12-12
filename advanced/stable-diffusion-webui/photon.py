from leptonai.photon import Photon


class WebUI(Photon):
    webui_version = "v1.6.0"
    cmd = [
        "bash",
        "-c",
        (
            "apt-get update && apt-get install -y wget libgoogle-perftools-dev && wget"
            f" -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/{webui_version}/webui.sh"
            " && chmod +x ./webui.sh && ACCELERATE=True ./webui.sh -f --listen --port"
            " 8080"
        ),
    ]
    deployment_template = {
        "resource_shape": "gpu.a10",
    }
