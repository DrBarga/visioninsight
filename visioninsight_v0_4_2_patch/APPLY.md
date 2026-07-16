# Apply VisionInsight v0.4.2 patch

1. Back up or commit your current repository.
2. Extract the ZIP next to your `visioninsight` folder.
3. Run the included installer from PowerShell:

```powershell
python .\visioninsight_v0_4_2_patch\apply_v0_4_2.py "C:\Users\bogda\OneDrive\Desktop\visioninsight"
cd "C:\Users\bogda\OneDrive\Desktop\visioninsight\backend"
python -m unittest discover -s tests -v
python -m uvicorn app.main:app --reload
```

The installer backs up replaced files under:

```text
.patch_backups/v0.4.2/
```

Recommended smoke tests in `/docs`:

1. `mode=fast`: no transcript, no CLIP, no output video, `frame_stride=2`.
2. `mode=balanced`: CV artifacts and output video, without Whisper or CLIP.
3. `mode=full`: complete CV + Whisper + CLIP pipeline.
4. Analyze two videos sequentially and confirm that both runs start people track IDs from `1`.
5. Inspect `summary.json -> timings` to compare the expensive stages.

Repository cleanup after confirming the patch:

```powershell
git rm -r --cached .idea 2>$null
git rm --cached screenshot_server_stdout.log 2>$null
git status
```

Do not delete your local `yolov8n.pt`; the new `.gitignore` only prevents it from being committed.
