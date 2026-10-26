FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN python -m pip install --no-cache-dir pydicom==3.0.1 scikit-learn==1.5.2 matplotlib==3.9.2
ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /workspace
CMD ["python", "scripts/check_local_global.py"]
