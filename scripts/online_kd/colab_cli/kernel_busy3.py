# Keeps the Colab kernel busy (status BUSY) for the whole session, so the runtime is not reclaimed as idle while
# downloads and training run outside the kernel. Exits once both lanes report DONE / FAILED, or after 10 h.
import os, time
LANES = ("kd_e4rk", "kd_e4qm", "kd_e4r101")
t0 = time.time()
while time.time() - t0 < 10 * 3600:
    st = {l: (open(f"/work/out/{l}.status").read().strip() if os.path.exists(f"/work/out/{l}.status") else "-") for l in LANES}
    if all(v.startswith(("DONE", "FAILED")) for v in st.values()):
        break
    print(time.strftime("%H:%M"), st, flush=True)
    time.sleep(300)
print("kernel_busy3 exit", st, flush=True)
