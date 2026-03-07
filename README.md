# CutoutCam

Webcam face-tracking avatar renderer with a South Park style character generator.

## What It Does

- Tracks face landmarks from webcam frames.
- Detects:
  - turn (`left/right/up/down/center`)
  - eye state
  - mouth state
  - brow state
  - head roll rotation
- Renders a layered PNG character.
- Streams the rendered frame to a virtual camera (`pyvirtualcam`, optional).
- Provides OpenCV sliders to tune character layout in real time.

## Run

```powershell
poetry install
poetry run python main.py
```

Alternative:

```powershell
.\scripts\dev\run_main.ps1
```

## Controls UI

Use `Prev` / `Next` buttons to switch control pages:

1. `Page 1: Character`
2. `Page 2: Face Features`
3. `Page 3: Background`
4. `Page 4: Mask`

Buttons:

- `Save` writes layout to `configs/characters/default_layout.json`
- `Pick BG` selects a custom background image
- `Clear BG` removes custom background
- `Quit` exits (same as `Esc`)

## Project Structure

```text
CutoutCam/
  main.py
  assets/
    characters/default/parts/
  configs/
    characters/default_layout.json
  opensee/
  cutoutcam/
    analyzers/
      face_analyzer.py
    generators/
      character_generator.py
      square_generator.py
      console_generator.py
    app_controls.py      # Layout UI, trackbars, save/load settings
    app_runtime.py       # Camera, virtual camera, state smoothing
    state.py
```

## Notes

- `pyvirtualcam` is optional. If unavailable, app still runs without virtual camera output.
- Character part PNGs are loaded from `assets/characters/<char_id>/parts`.
