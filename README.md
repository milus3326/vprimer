## Installation

Requires **Python ≥ 3.14** and **git**

1. Clone the repository:
  ```bash
   git clone https://github.com/milus3326/vprimer.git
   cd vprimer
  ```
2. Create and activate a virtual environment:
  ```bash
   python -m venv .venv
  ```
  - Windows: `.\.venv\Scripts\Activate.ps1`
  - Linux / macOS: `source .venv/bin/activate`
3. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
4. Install **MAFFT** ([https://mafft.cbrc.jp/alignment/software/](https://mafft.cbrc.jp/alignment/software/)), either:
  - add a `mafft` executable to your `PATH`, **or**
  - place the MAFFT folder in the same directory.
5. Run the app:
  ```bash
   python.exe run.py
  ```

