from mnn_engine import v2_inference
from mnn_expr import v3_inference
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    num_inferences = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if model_path.endswith('.mnn'):
        v2_inference(model_path, num_inferences)
        v3_inference(model_path, num_inferences)