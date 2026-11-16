Exploratory Data Analysis (EDA) – Titanic Dataset

Objective
Extract insights from the Titanic passenger dataset using statistical and visual exploration to find patterns, trends and anomalies.

Tools
Python · Pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook

Repository contents
| File | Description |
|---|---|
| `Task5_Titanic_EDA.ipynb` | Fully executed Jupyter notebook (code + outputs + observations) |
| `Task5_EDA_Report.pdf` | PDF report of findings |
| `titanic.csv` | Dataset (891 rows × 15 columns, from seaborn-data) |
| `build_nb.py`, `make_pdf.py` | Scripts used to generate the notebook and the report |

What I did
1. Inspection – `.info()`, `.describe()`, `.value_counts()`, shape, dtypes.
2. Data quality – missing values (deck 77%, age 20%), duplicates; imputed age by class/sex median, embarked by mode; engineered `family_size`, `has_cabin`, `age_group`.
3. Univariate – histograms/KDE, count plots.
4. Bivariate– survival-rate bar plots, boxplots, violin/KDE, scatter plots.
5. Multivariate – correlation heatmap, pairplot, class × sex × survival heatmap.
6. Outliers & skewness – IQR rule on fare, log1p transform (skew 4.8 → 0.4).
7. Observations written under every visual, a summary table of findings, and answers to the interview questions.

 Key findings
- Overall survival: 38.4%.
- Sex: women 74% vs men 19%.
- Class: 63% / 47% / 24% for 1st / 2nd / 3rd. 1st-class women 97% vs 3rd-class men 13.5%.
- Children (0–12) survived at 58%; seniors 23%.
- Higher fare and having a recorded cabin → higher survival.
- Small families (2–4) did best; solo travellers 30%.
- `pclass` and `fare` are strongly correlated (−0.55) → multicollinearity risk.

How to run
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook Task5_Titanic_EDA.ipynb
```
