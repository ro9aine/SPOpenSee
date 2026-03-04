# SPOpenSee

South Park face tracking.

## Project Structure

```text
SPOpenSee/
  main.py                        # Local dev entrypoint
  assets/
    characters/
      default/
        parts/                   # PNG layers and parts
  configs/
    characters/
      default.json               # Character metadata and asset paths
  scripts/
    dev/
      run_main.ps1               # Dev runner
  tests/
  opensee/                       # Tracker/runtime dependencies
  spopensee/
    analyzers/
      face_analyzer.py           # Canonical analyzer import path
    generators/
      square_generator.py        # Canonical square generator import path
      console_generator.py
      character_generator.py
    analizer.py                  # Legacy module (kept for compatibility)
    charactergenerator.py        # Legacy module (kept for compatibility)
    state.py
```

## Conventions

- Put runtime code in `spopensee/` subpackages.
- Put image assets in `assets/`.
- Put JSON configs in `configs/`.
- Keep `main.py` minimal and focused on wiring.
- Keep legacy modules only as compatibility shims while migrating imports.
