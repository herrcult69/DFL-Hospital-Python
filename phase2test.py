# run this once as a setup script: dfl/create_test_models.py
from pathlib import Path

nodes = [
    "127.0.0.1:8000:9000",   # bootstrap
    "127.0.0.1:8001:9001",   # worker 1
    "127.0.0.1:8002:9002",   # worker 2
]
round_ = 1

Path("./models").mkdir(exist_ok=True)

for node_id in nodes:
    safe = node_id.replace(":", "_")
    path = Path(f"./models/round{round_}_{safe}.safetensors")
    # 10 MB of dummy data
    path.write_bytes(b"x" * 10 * 1024 * 1024)
    print(f"Created {path}")