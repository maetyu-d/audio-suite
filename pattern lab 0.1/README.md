# Pattern Lab

Single-pattern version (pre-layers) with an always-enabled code editor:
- Built-in generators live in `app/patterns.py`
- Script generator loads external `.py` via presets (`generator`: "Python Script")

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Editing code
Use the **Edit Pattern Code…** button:
- If current generator is **Python Script** and `script_path` is set, it opens that file.
- Otherwise it opens `app/patterns.py` and hot-reloads it on save.
