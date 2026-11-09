# Model files

The old placeholder entries named `det_10g.onnx`, `w600k_r50.onnx` and `face_landmarker.task` were removed from this branch after the repository audit showed that they were 51-byte `/api/documents/...` text references, not model weights.

**No usable model weights are currently stored in this directory.**

Run the asset checker to confirm the current state:

```bash
python3 tools/validate_assets.py
```

## Installing local model assets

1. Select a Face Swap model whose **weights** license permits the intended private use.
2. Confirm the model can run on iOS and that its input/output tensor contract is known.
3. Convert and validate the model for Core ML on a development Mac.
4. Keep private/licensed weights outside this public repository, for example under the ignored local path `ios/RetakeApp/Resources/Models/`.
5. Import the compiled `.mlmodelc` directory in the iOS app, or bundle a validated model in a local Xcode build.
6. Record the exact model name, version, source, checksum, license and conversion settings in a local build manifest.

A model file existing on disk does not prove that it loads, executes on Neural Engine, or produces a valid face-swap result. Those checks must run on the target iPhone.

The repository ignores local model weight formats so they are not accidentally published to this public project.