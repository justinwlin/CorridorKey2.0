# Running CorridorKey on cloud GPUs

`remote/run.py` runs CorridorKey on GPUs rented from a cloud provider, for when
you don't have an NVIDIA GPU or want to process a folder of clips on several
GPUs at once. It runs on macOS, Linux, and Windows and needs only Python 3.10+
and the `ssh` client. No local GPU, PyTorch, or model weights are needed.

For each run it:

1. Rents one or more GPU machines.
2. Uploads this repository's code and your clips over SSH.
3. Installs the requirements and downloads the weights on each machine.
4. Runs `infer.py` on each clip and downloads the outputs.
5. Releases the machines, including when the run fails or you press Ctrl-C.

Runpod is the first supported provider. Others can be added without changing
the runner, see [Adding a provider](#adding-a-provider).

## Runpod

Create an API key at
[console.runpod.io/user/settings](https://console.runpod.io/user/settings),
then:

```
export RUNPOD_API_KEY=your_key
python remote/run.py --input shot.mp4 --output_dir out
```

Machines bill per second to your Runpod account while they run. The script
prints each pod's hourly price when it starts it.

Every flag that `run.py` does not define is passed to `infer.py` unchanged, so
the examples in the main README work as they are:

```
python remote/run.py \
  --input shot.mp4 \
  --output_dir out_frame1 \
  --hint_first_frame matte_frame_0001.png \
  --hint_quality 0.95 \
  --outputs alpha fg cutout
```

`--hint_first_frame` and `--hint_video` take local paths and are uploaded with
the clip.

### Several clips on several GPUs

Point `--input` at a folder of clips and choose how many machines to rent:

```
python remote/run.py --input clips/ --output_dir out --machines 3
```

The folder is read the same way as with the drag-and-drop launcher: video
files and `Input`/`AlphaHint` frame folders are clips, companion hint files
(`<clip>FrameHint.png`, `<clip>_AlphaHint.mp4`) are applied automatically, and
each clip's outputs go to `out/out_<clip>`. Each machine takes the next
unprocessed clip until none are left. A single clip always runs on one machine,
because splitting a clip would break the temporal context between its frames.

### Runpod options

| Flag | Default | Purpose |
|---|---|---|
| `--runpod_gpu` | L40S, RTX 6000 Ada, RTX A6000, L40, A40, RTX 4090 | GPU types to try, in order. Repeat the flag to give your own list. |
| `--runpod_network_volume` | none | Network volume to keep the weights on between runs. |
| `--runpod_volume_gb` | 60 | Pod disk for weights, inputs, and outputs when there is no network volume. |
| `--runpod_cloud_type` | `SECURE` | `COMMUNITY` is cheaper, with less availability. |
| `--runpod_image` | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` | Container image. |

The default GPUs have 48 GB, which lets CorridorKey use its largest temporal
window up to 1440p (see [Long Clips And Low VRAM](../README.md#long-clips-and-low-vram)).

Without a network volume, each machine downloads the ~5 GB of weights at the
start of the run. With `--runpod_network_volume`, they are downloaded once and
reused by later runs and by every machine in a run. Create the volume in the
Runpod console; pods then start in that volume's data center.

### Other options

- `--keep_machines` leaves the machines running after the run and prints the
  SSH key and commands to connect. They keep billing until you delete them in
  the Runpod console.
- `--ready_timeout` is how long to wait for a machine to accept SSH (default
  900 seconds).

## Adding a provider

Any cloud that rents a CUDA Linux machine with SSH access can be added in one
file of about 100 lines. See [ADDING_A_PROVIDER.md](ADDING_A_PROVIDER.md) for
the interface, the machine requirements, and a testing checklist.

## Tests

```
python remote/tests/test_run.py
```

The tests use a fake provider, so they need no cloud account or GPU.
