# Single-stage Dockerfile for Government Document Verification System
# NOTE: Python 3.12 for PaddlePaddle compatibility (paddlepaddle==2.6.2 requires numpy<2.0)
#
# Deliberately SINGLE-stage. A previous multi-stage build copied the whole
# site-packages tree between stages and produced non-deterministic
# "zlib.error: Error -2 while decompressing data" failures on import of
# compiled extensions (hit pyclipper on some builds, scipy on others).
# Installing directly into the final image removes that copy entirely.

FROM python:3.12-bookworm

WORKDIR /app

# Build toolchain + PaddleOCR runtime libraries in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# llama-cpp-python must come from the prebuilt CPU wheel index, otherwise it
# compiles from source via CMake and can hang for over an hour.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        -r requirements.txt

# paddleocr transitively pulls opencv-python AND opencv-contrib-python
# alongside the headless build we actually want. All three install into the
# same cv2 namespace and can conflict. Keep only headless.
RUN pip uninstall -y opencv-python opencv-contrib-python && \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.6.0.66

# Fail the BUILD if any compiled extension is corrupt, rather than discovering
# it at container run time. This also tells us whether corruption exists at
# install time or is introduced later by the image layer export.
RUN python - <<'PY'
import sys
mods = ["numpy", "zlib", "cv2", "scipy", "scipy._lib._ccallback", "pyclipper",
        "skimage.morphology", "paddle", "paddleocr", "llama_cpp"]
failed = []
for m in mods:
    try:
        __import__(m)
        print(f"OK   {m}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
        failed.append(m)
if failed:
    print("BUILD-TIME IMPORT FAILURES:", failed)
    sys.exit(1)
print("all imports OK at build time")
PY

COPY . .

RUN mkdir -p data logs models

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# urllib is stdlib; raises on non-2xx so an unhealthy app actually reports unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=5).status==200 else 1)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
