"""
Quick parser verification.
"""
from pathlib import Path
import tempfile
from src.parsers.yaml_parser import parse_file as parse_yaml
from src.parsers.python_parser import parse_file as parse_py

# Test YAML parser with a sample K8s manifest
yaml_content = '''
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: katib-controller
  namespace: kubeflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: katib
  template:
    spec:
      containers:
      - name: katib-controller
        image: docker.io/kubeflowkatib/katib-controller
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: katib-controller-svc
  namespace: kubeflow
spec:
  selector:
    app: katib
  ports:
  - port: 8080
'''

py_content = '''
def hello_world(name: str) -> None:
    """Print a greeting."""
    print(f"Hello, {name}!")

class Greeter:
    def __init__(self, greeting: str = "Hello"):
        self.greeting = greeting

    def greet(self, name: str) -> str:
        """Return a greeting string."""
        return f"{self.greeting}, {name}!"

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

with tempfile.TemporaryDirectory() as tmpdir:
    yaml_path = Path(tmpdir) / "test.yaml"
    yaml_path.write_text(yaml_content)

    py_path = Path(tmpdir) / "test.py"
    py_path.write_text(py_content)

    print("=== YAML Parser ===")
    for chunk in parse_yaml(yaml_path):
        print(f"  chunk_id: {chunk['chunk_id']}")
        print(f"  content_text (first 200): {chunk['content_text'][:200]}")
        print(f"  source_type: {chunk['source_type']}")
        print()

    print("=== Python Parser ===")
    for chunk in parse_py(py_path):
        print(f"  chunk_id: {chunk['chunk_id']}")
        print(f"  content_text (first 200): {chunk['content_text'][:200]}")
        print()
