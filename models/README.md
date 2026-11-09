# Model files

The files currently present in this directory are not validated model binaries. The repository audit found 51-byte entries that resolve to `/api/documents/...` reference text, not usable ONNX or MediaPipe model weights.

Do not treat these files as installed models. Run:

```bash
python3 tools/validate_assets.py
```

## Installing local model assets

1. Select a Face Swap model whose **weights** license permits the intended private use.
2. Confirm the model can run on iOS and that its input/output tensor contract is known.
3. Convert and validate the model for Core ML on a development Mac.
4. Keep private/licensed weights outside this public repository, for example under the ignored local path `ios/Retake/Resources/Models/`.
5. Record the exact model name, version, source, checksum, license and conversion settings in a local build manifest.

A model file existing on disk does not prove that it loads, executes on Neural Engine, or produces a valid face-swap result. Those checks must run on the target iPhone.
