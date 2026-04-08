import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables or defaults

SOULX_PYTHON = os.environ.get(
    "SOULX_PYTHON",
    r"C:\Users\archi\miniconda3\envs\soulxsinger\python.exe",
)

UTAU_GENERATE_DLL = os.environ.get(
    "UTAU_GENERATE_DLL",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "API", "UtauGenerate", "bin", "Debug", "net9.0", "UtauGenerate.dll")),
)
