FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/dmx

COPY . .

RUN pip install --no-cache-dir . && \
    rm -rf /opt/dmx/.git /opt/dmx/build /opt/dmx/dist /opt/dmx/*.egg-info

HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD ["dmx", "--help"]

ENTRYPOINT ["dmx"]
