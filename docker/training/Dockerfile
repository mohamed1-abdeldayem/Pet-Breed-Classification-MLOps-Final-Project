FROM pet-breed

WORKDIR /workspace

COPY training ./training
COPY configs ./configs

ENV PET_LABEL_MAP=/workspace/configs/label_map.json

CMD ["python", "/workspace/training/train.py", "--help"]
