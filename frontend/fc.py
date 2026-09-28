import pathlib
p = pathlib.Path("src/services/downscalingApi.ts")
t = p.read_text(encoding="utf-8")
# The previous file's header/body was replaced; strip the leftover GNN client.
start = t.index("/**")
gnn = t.find("import { useCallback")
print("has leftover gnn client:", gnn != -1)
