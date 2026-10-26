FROM python:3.11-slim-bookworm
WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir '.[test]'
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["python", "-m", "pytest", "-q"]
