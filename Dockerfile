# Optional: run the notebook end-to-end offline in a reproducible container.
#   docker build -t crypto-fund .
#   docker run --rm crypto-fund
FROM python:3.11-slim

# uv for fast, locked dependency installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy the whole project first: the editable `fund` package must be present for
# `uv sync` to build it. Data CSVs are baked in so the run needs no network.
COPY . .

# Install the exact locked versions (fails if uv.lock is out of date).
RUN uv sync --frozen

# Execute the notebook top-to-bottom (regenerates all outputs).
CMD ["uv", "run", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "notebook.ipynb"]
