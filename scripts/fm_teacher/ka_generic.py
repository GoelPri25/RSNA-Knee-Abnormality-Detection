import pathlib, time
p = pathlib.Path("/content/fm/STATUS")
while True:
    s = p.read_text().strip() if p.exists() else "setup"
    print(time.strftime("%H:%M"), s[:200], flush=True)
    if s.startswith(("ALLDONE", "FAILED")):
        break
    time.sleep(60)
