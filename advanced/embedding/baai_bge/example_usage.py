import subprocess
import time
import socket

from leptonai.client import Client, local, current  # noqa: F401


def is_port_open(host, port):
    """Check if a port is open on a given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except socket.error:
            return False


def wait_for_port(host, port, interval=5):
    """Wait for a port to be connectable."""
    while True:
        if is_port_open(host, port):
            print(f"Port {port} on {host} is now connectable!")
            break
        else:
            print(
                f"Port {port} on {host} is not ready yet. Retrying in"
                f" {interval} seconds..."
            )
            time.sleep(interval)


def main():
    # launches "python main.py" in a subprocess so we can use the client
    # to test it.
    #
    print("Launching the photon in a subprocess on port 8080...")
    p = subprocess.Popen(
        ["python", "main.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    wait_for_port("localhost", 8080)

    # Note: this is not necessary if you are running the photon in the lepton
    # server. To run it in the server, you can do
    #   lep photon run -n bge -m main.py --resource-shape gpu.a10
    # and then instead of using local, you can use the client as
    #   c = Client(current(), "bge")
    # where current() is a helper function to get the current workspace.

    c = Client(local())
    # c = Client(current(), "bge")
    print("\nThe client has the following endpoints:")
    print(c.paths())
    print("For the encode endpoint, the docstring is as follows:")
    print("***begin docstring***")
    print(c.encode.__doc__)
    print("***end docstring***")

    print("\n\nRunning the encode endpoint...")
    query = "The quick brown fox jumps over the lazy dog."
    ret = c.encode(sentences=query)
    print("The result is (truncated, showing first 5):")
    print(ret[:5])
    print(f"(the full result is a list of {len(ret)} floats)")

    print("\n\nRunning the rank endpoint...")
    sentences = [
        "the fox jumps over the dog",
        "the photon is a particle and a wave",
        "let the record show that the shipment has arrived",
        "the cat jumps on the fox",
    ]
    rank, score = c.rank(query=query, sentences=sentences)
    print("The rank and score are respectively:")
    print([(r, s) for r, s in zip(rank, score)])
    print(f"The query is: {query}")
    print("The sentences, ordered from closest to furthest, are:")
    print([sentences[i] for i in rank])

    print("Finished. Closing everything.")
    # Closes the subprocess
    p.terminate()


if __name__ == "__main__":
    main()
