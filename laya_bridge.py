import json
import sys

import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    try:
        result = agent.predict(request["state"], request["questions"])
    except Exception as error:
        print(json.dumps({"id": request.get("id"), "error": str(error)}), flush=True)
        continue
    result["id"] = request.get("id")
    print(json.dumps(result), flush=True)
