# Keeps the Colab kernel busy (execution_state "busy") while training runs outside it, so the runtime is not
# reclaimed as idle. Exits when /work/out/kd_e4on.status says DONE / FAILED, or after the exec timeout.
import time, os
t0 = time.time()
while time.time() - t0 < 6 * 3600:
    st = open("/work/out/kd_e4on.status").read().strip() if os.path.exists("/work/out/kd_e4on.status") else "PREPARING"
    if st.startswith(("DONE", "FAILED")) and time.time() - t0 > 600:
        break
    print(time.strftime("%H:%M"), st, flush=True)
    time.sleep(300)
print("kernel_busy exit", flush=True)
