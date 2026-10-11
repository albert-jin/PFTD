FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==1.26.4 sympy==1.12
WORKDIR /opt/worker
COPY pftd/__init__.py pftd/worker.py pftd/codec.py pftd/contracts.py ./pftd/
USER 65534:65534
ENTRYPOINT ["python", "-m", "pftd.worker"]
