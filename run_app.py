import uvicorn
import spaces
from app.main import app

@spaces.GPU
def _dummy_gpu_function():
    pass

if __name__ == "__main__":
    # Hugging Face Spaces expects the application to run on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
