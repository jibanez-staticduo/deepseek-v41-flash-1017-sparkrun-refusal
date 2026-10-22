# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY overlay/vllm/ /opt/spark3/candidate/vllm/vllm/
COPY payload/ /opt/spark3/candidate/vllm/vllm/
COPY assets/refusal_dirs_1017.safetensors /opt/sparkrun-refusal/refusal_dirs_1017.safetensors
COPY scripts/launch.py /opt/sparkrun-refusal/launch.py
COPY runtime.json /opt/sparkrun-refusal/runtime.json
COPY manifest.json /opt/sparkrun-refusal/manifest.json
COPY notices/ /opt/sparkrun-refusal/notices/
LABEL org.opencontainers.image.title="DeepSeek 1017 TP3 SparkRun refusal" \
      org.opencontainers.image.source="https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal"
ENTRYPOINT []
CMD ["bash"]
