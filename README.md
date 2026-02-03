# Sat-U-Net

Semantic segmentation notebooks and production notes for the DeepGlobe land-cover dataset.

**Repo Setup**
1. `git clone https://github.com/GaneshMakkena/Sat-U-Net.git`
2. `cd Sat-U-Net`

**Environment Setup**
1. `python3 -m venv .venv`
2. Activate the environment: macOS/Linux `source .venv/bin/activate` Windows `\.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Optional for notebooks: `pip install jupyter`

**Dataset Layout**
The dataset is not stored in this repo due to GitHub size limits. Place your files locally in this layout:

```text
Sat-U-Net/
  train/
    *_sat.jpg
    *_mask.png
  valid/
    *_sat.jpg
    *_mask.png (optional)
  test/
    *_sat.jpg
  metadata.csv
  class_dict.csv
```

**Run**
1. Open `Sat_U_Net_Production.ipynb` or `Sat_IMG_U_Net.ipynb` in Jupyter.
2. Update any dataset paths inside the notebook as needed.
3. See `docs/PRODUCTION.md` for the production-grade workflow.
