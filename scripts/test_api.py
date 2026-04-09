import json
import urllib.request
from pathlib import Path


def main() -> None:
    payload_path = Path("examples/example_request.json")
    with open(payload_path, "r") as f:
        payload = json.load(f)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        print(body)


if __name__ == "__main__":
    main()
