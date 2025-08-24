import subprocess
import os

def main():
    # Find the path to the Streamlit app bundled in the package
    app_path = os.path.join(os.path.dirname(__file__), "app", "trading_assistant.py")

    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Could not find Streamlit app at {app_path}")

    # Launch Streamlit
    subprocess.run(["streamlit", "run", app_path])
