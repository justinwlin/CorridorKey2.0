# Adding a GPU provider

`remote/run.py` can run CorridorKey on any cloud that rents a CUDA Linux
machine reachable over SSH. A provider is one Python file of about 100 lines.
It only starts a machine, reports its SSH address, and shuts it down. The
shared runner does the rest over SSH: uploading code and clips, setup,
inference, downloads, spreading clips across machines, and cleanup.

`remote/providers/runpod.py` is a complete example to copy.

## How a run works

For each machine, the runner:

1. Calls `create(name, public_key)` with a new SSH public key for this run.
2. Calls `wait_for_ssh(instance_id, timeout)` and retries SSH until it
   connects.
3. Uploads the code to `<workdir>/corridorkey` and runs
   `remote/setup_machine.sh`. The script installs `ffmpeg` and the Python
   requirements and downloads the weights.
4. Runs `infer.py` on each clip it takes from the shared queue. Inputs and
   outputs go through `<workdir>/corridorkey_jobs/<run>/<clip>`.
5. Calls `destroy(instance_id)` when the queue is empty, when the machine
   fails, or when the user presses Ctrl-C.

## What the machine must provide

- An NVIDIA GPU (RTX 30-series or newer) and a driver that supports
  CUDA 12.8.
- `python` and `pip` on the `PATH`, with a CUDA build of PyTorch installed.
  The README was tested with torch 2.9.1.
- SSH as a user that can run `apt-get` (normally `root`), accepting the public
  key passed to `create`. Most providers take the key as an API field, an
  environment variable, or cloud-init user data.
- `bash`, `tar`, and `flock`. Ubuntu images have all three.
- About 10 GB free under `workdir` for weights, plus room for clips and
  outputs. EXR output and `--low_vram` need much more.

If the provider's images don't meet these, choose a different image in
`create` instead of changing `setup_machine.sh`.

## Steps

### 1. Write the provider

Create `remote/providers/<name>.py`:

```python
from __future__ import annotations

import os

from .base import Provider, SSHTarget


class ExampleProvider(Provider):
    name = "example"          # value of --provider
    workdir = "/workspace"    # persistent disk if the provider has one

    @classmethod
    def add_arguments(cls, group):
        # Prefix every flag with the provider name so flags never collide.
        group.add_argument("--example_gpu", default="L40S")

    def __init__(self, args):
        super().__init__(args)
        self.token = os.environ.get("EXAMPLE_API_KEY")
        if not self.token:
            raise RuntimeError("Set EXAMPLE_API_KEY.")

    def create(self, name, public_key):
        # Start one machine with one GPU that accepts public_key.
        # Return the provider's id for the machine. Print its price if the API
        # returns one, so users can see what they are paying.
        ...

    def wait_for_ssh(self, instance_id, timeout):
        # Poll until the machine has a public address and SSH port, then
        # return SSHTarget(host, port, user). Raise if it fails or times out.
        ...

    def destroy(self, instance_id):
        # Terminate the machine, not just stop it, so storage stops billing.
        # Treat "already gone" as success, because destroy can run twice.
        ...
```

### 2. Register it

In `remote/providers/__init__.py`:

```python
from .example import ExampleProvider

PROVIDERS = {
    RunpodProvider.name: RunpodProvider,
    ExampleProvider.name: ExampleProvider,
}
```

Users then run `python remote/run.py --provider example ...`.

### 3. Document it

Add a section for the provider to `remote/README.md`: the environment
variable for the key, its flags with their defaults, and anything that costs
money besides GPU time, such as storage that outlives the machine.

## Rules

- **Use only the standard library.** The runner works on a laptop with no
  CorridorKey requirements installed. Call the API with `urllib.request` (see
  `RunpodProvider._request`) rather than adding a vendor SDK.
- **Always release machines.** `destroy` must work no matter what state the
  machine is in, and must not raise when the machine is already gone. The
  runner calls it from several exit paths.
- **Don't block forever.** `wait_for_ssh` must honour `timeout` and raise if
  the machine stops while it waits.
- **Put errors in the exception message.** Include the API's response body,
  for example "no L40S available in this region", so users can act on it.
- **Keep credentials in environment variables.** Never take them as flags,
  because flags show up in shell history and process lists.
- **Set a User-Agent.** Some provider APIs reject Python's default one. Runpod
  returns HTTP 403 (Cloudflare error 1010).

## Testing

### Offline

`remote/tests/test_run.py` runs the whole runner against a fake provider and
a fake SSH layer:

```
python remote/tests/test_run.py
```

Run it after changing anything in `remote/`. Its `FakeProvider` is also a
minimal working example of the interface.

### Against the real provider

Use a short clip, and keep `--num_frames` small so the run is quick:

```
export EXAMPLE_API_KEY=...
python remote/run.py --provider example \
  --input clip.mp4 --output_dir out_test --num_frames 24
```

Check each of these:

1. The run finishes, and `out_test` holds `alpha` and `fg` outputs that play.
2. The provider's dashboard shows no running machines afterwards.
3. Press Ctrl-C while setup is running. The run prints "releasing machines",
   and the dashboard shows the machine terminated.
4. A folder of three clips with `--machines 2` starts two machines and
   produces three `out_<clip>` folders.
5. An unavailable GPU type fails with a readable message, and no machine is
   left behind.
6. `--keep_machines` leaves the machine running, and the printed SSH command
   connects to it. Delete the machine afterwards.

Put the results of these checks in the pull request.
