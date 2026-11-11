## Installation

Works on Windows, macOS and Linux. Needs **Python 3.14+**.

1. Clone the repository:
   ```bash
   git clone https://github.com/milus3326/vprimer.git
   cd vprimer
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   - Windows: `.venv\Scripts\activate`
   - macOS / Linux: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install **MAFFT** ([https://mafft.cbrc.jp/alignment/software/](https://mafft.cbrc.jp/alignment/software/)).
   Either:
   - put a `mafft` executable on your `PATH`:
     - macOS: `brew install mafft`
     - Linux: `sudo apt install mafft`
   - **or** unpack the portable package for your system next to `run.py`
5. Run the app:
   ```bash
   python run.py
   ```
   The first time, enter your email (NCBI requires one) and an [NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/)
