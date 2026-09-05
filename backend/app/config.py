from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT.parent / "data" / "raw"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "riskgraph_download" / "riskgraph_final_model"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "riskgraph_download" / "riskgraph_outputs"))
PREDICTION_TIME = os.getenv("PREDICTION_TIME", "2025-12-01T00:00:00")
