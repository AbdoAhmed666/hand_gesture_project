# ✅ main.py النهائي
from src.deploy.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
