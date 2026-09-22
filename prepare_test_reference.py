"""Fetch and verify the pinned Tencent source used by test_adapter.py.

The downloaded source is kept locally under reference/ and is not redistributed
in this repository. This command downloads source only, not model weights.
"""

import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "c8ffd07206f1b843697606968196e8f59f8ff38c"
FILENAME = "modeling_hunyuan_image_3.py"
SHA256 = "0dd3ec2592ab7458534a6b22eb0c16864aaee9c1869e4a1422ab5e308e02a71b"
URL = f"https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil/resolve/{REVISION}/{FILENAME}"


def main():
    destination = Path(__file__).parent / "reference" / FILENAME
    if destination.exists():
        data = destination.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError(f"SHA256 mismatch for {destination}; expected Tencent revision {REVISION}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(data)
    print(f"Verified {destination.name} at revision {REVISION}")


if __name__ == "__main__":
    main()
