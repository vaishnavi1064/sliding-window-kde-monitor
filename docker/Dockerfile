FROM python:3.12-slim

WORKDIR /app

# Copy only what pip needs first so dependency layers cache across code edits.
COPY pyproject.toml setup.py README.md ./
COPY sketch/ ./sketch/
COPY src/ ./src/

# g++ is here to compile the native core (Phase 5). Without it the extension is
# skipped -- it is declared `optional` -- and SKETCH_BACKEND=auto would quietly
# fall back to the Python core, so the deployed pipeline would run ~15x slower
# than the benchmark it is documented against. SWAKDE_REQUIRE_NATIVE makes a
# compile failure fail the image build instead of producing that surprise.
#
# The compiler is then removed again: it is a build dependency, not a runtime
# one. libstdc++6 is installed explicitly so that marking it manual keeps
# `autoremove` from taking it away with g++ -- the extension links against it.
# The import check runs *after* the purge, deliberately: that is the only
# ordering that proves the runtime image works rather than the build stage, so
# an autoremove that took libstdc++6 with it fails the build here instead of
# producing an image that dies on container start.
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ libstdc++6 \
 && SWAKDE_REQUIRE_NATIVE=1 pip install --no-cache-dir -e ".[streaming]" \
 && apt-get purge -y g++ \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && python -c "import sketch.native as n; assert n.AVAILABLE, n.IMPORT_ERROR; print('native core:', n.compiler())"

COPY streaming/ ./streaming/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "streaming.consumer"]
