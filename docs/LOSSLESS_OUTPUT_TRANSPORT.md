# Retake — native-size quality checks and lossless output transport

## Reproduction
A real user submitted a photo that was already in their reference bank. The v22 engine reported original and output ArcFace 0.915, five verified rounds, 633.6 seconds. The downloaded JPEG showed white, black and colored curves around the head and jacket. The original was clean. The logs repeatedly said `Kontur taraması ... not_examined` and the diffusion worker reported that target and result dimensions differed. These are failure signals, not evidence of a good image.

## Root causes
- FaceFusion requires even input dimensions. The v22 pipeline padded the only target file, then cropped composites to the native dimensions. Realism, rim diagnostics and regional detail comparisons subsequently compared different sizes.
- A preblended RGB result is already feathered. Its alpha channel is an edit-extent matte, not opacity. A browser canvas premultiplies alpha; at alpha=1/255 RGB can round to 0 or 255 per channel. Copying those decoded colors as opaque creates bright, black and colored outlines even if the server-side composite was clean. A dark-rim detector alone cannot detect this later browser-side corruption.

## Fix
- Keep the original target file and its native dimensions unchanged. Use a separate lossless edge-replicated PNG for odd-sized FaceFusion inference only.
- Keep the original continuous alpha during compositing, geometry, ArcFace checks and rim cleanup. At delivery only, encode the already-composited RGB with alpha 255 for changed pixels within the matte, 0 otherwise.
- The server advertises `lossless_preblended_transport: true` on /health and `X-Retake-Alpha-Transport: binary` on result responses.
- The web application rejects legacy fractional-alpha preblended payloads, refuses a mismatching native render size, and scales opaque colors and binary extent separately when a very large input requires resizing.
- The existing v22 profile, 25-reference contract, ArcFace tolerances, Ultra schedule, references and model weights are retained. The change does not train a model or promise a better biometric score.
- Unit tests simulate browser premultiply/unpremultiply and native odd/even dimensions; mocked pipeline regression checks actual result bytes and original pixels outside the extent.

## Local update and verification
Stop only after active processing completes. From the existing worktree:
```powershell
git pull --ff-only origin main
.\tools\update_retake_ultra.ps1 -SkipPull -SkipDeploy
```
Use another PowerShell window to check:
```powershell
(Invoke-RestMethod "http://127.0.0.1:8765/health").lossless_preblended_transport
```
Expected: True. Refresh the live web page. Rerun the same input, check that the downloaded image has no colored/black/white boundary and that the contour and regional detail gates actually run. Realism can still fail or be rejected for independent reasons (models, GPU resources, or identity). An already-matching reference photo is a strong negative control: don't mistake its initial high ArcFace for improved realism or interpret it as a percentage probability.

The user's source and result images must not be added to a public repository. Regression fixtures are synthetic.
