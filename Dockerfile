FROM python:3.12-slim
ARG task_type

ENV TASK_TYPE=$task_type
ENV EXECUTE_IN_DOCKER=1

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Copy and install requirements first (this layer will be cached)
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Copy Python files last (these change most frequently)
COPY --chown=algorithm:algorithm utils /opt/algorithm/utils/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm base_algorithm.py /opt/algorithm/
COPY --chown=algorithm:algorithm exp_configs /opt/algorithm/exp_configs/
COPY --chown=algorithm:algorithm some_checkpoints /opt/algorithm/some_checkpoints/

ENTRYPOINT python -m process $0 $@
