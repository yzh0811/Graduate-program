Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (!(Test-Path ".\.venv")) {
  python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

if (!(Test-Path ".\.env")) {
  Copy-Item "env.example" ".env"
}

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

