# ALGIONS AI LABS — Submission Form Guide
**Assessment:** Bus Passenger-Demand Forecasting  
**Role:** AI & Machine Learning  

Use this guide to copy and paste required entries directly into the Algions AI Labs individualized submission form.

---

### Section 1: Candidate Identification
| Form Field | Suggested Entry |
| :--- | :--- |
| **Full Name** | *[Enter your full legal name]* |
| **Registration Number** | *[Enter your student / registration number]* |
| **University Email** | *[Enter your institutional / university email address]* |

---

### Section 2: Assignment Confirmation
| Form Field | Entry |
| :--- | :--- |
| **Internship Role** | `AI & Machine Learning` |
| **Assigned Project Title** | `Bus Passenger-Demand Forecasting` |

---

### Section 3: Code and Version
| Form Field | Entry |
| :--- | :--- |
| **Repository URL** | `https://github.com/<your-username>/bus-passenger-demand-forecasting` *(push local repo to your GitHub)* |
| **Primary Branch** | `main` (or `master`) |
| **Final Commit ID** | `672a827` *(or latest commit hash)* |
| **Reviewer Access** | Ensure the repository is Public, or invite Algions reviewers if kept Private. |

---

### Section 4: Demonstration
| Form Field | Entry |
| :--- | :--- |
| **3–5 Minute Video Link** | *[Loom / YouTube / Google Drive unlisted link showing EDA, code structure, test suite passing, and real inference in Streamlit dashboard]* |
| **Live Demo URL** | Declared as `Local-only interactive Streamlit app and FastAPI service` (or provide deployed URL if hosted on Streamlit Community Cloud / Render / Railway). |
| **Local Demo Command** | `streamlit run app/dashboard.py` and `uvicorn app.api:app --reload` |

#### Video Walkthrough Script (3-5 Minutes):
1. **0:00–0:45 (Problem & Scope):** Introduce the business decision at D-1 18:00 cut-off (24h before trip departure), unit of prediction $(Route, Departure\_Datetime)$, and target `passenger_count`.
2. **0:45–1:45 (Data & Leakage Prevention):** Show the 17,425 trip records across 6 routes. Explain the strict chronological split (Jan–Oct train, Oct–Dec val, Dec–Mar held-out test) and why 7d/14d lags prevent future leakage.
3. **1:45–2:45 (Model Comparison & Metrics):** Show `artifacts/metrics_summary.json` and `reports/figures/model_comparison_bar.png`. Highlight the 67.8% relative error reduction from Historical Baseline (27.64% WAPE) to LightGBM (8.89% WAPE, MAE 2.63 pax).
4. **2:45–4:00 (Live Interface Demo):** Open Streamlit app (`streamlit run app/dashboard.py`), select `ROUTE_101` at 08:00, show real dynamic forecast (47.7 pax), 95% CI, status code (`NEAR_CAPACITY_MONITOR`), and simulate what happens when snow storm or event occurs.
5. **4:00–4:30 (Responsible AI & Tests):** Run `pytest tests/ -v` showing all 16 tests passing, and explain human-in-the-loop dispatch authority.

---

### Section 5: Documentation
| Form Field | Entry |
| :--- | :--- |
| **Technical Report Link** | `reports/TECHNICAL_REPORT.md` (or GitHub raw/rendered URL) |
| **README Link** | `README.md` |
| **Architecture / Flow** | Documented in `README.md` Section 3 & `reports/TECHNICAL_REPORT.md` Section 3. |

---

### Section 6: Technology and Limitations
| Form Field | Entry |
| :--- | :--- |
| **Technology Stack** | Python 3.12, LightGBM, Scikit-learn, Pandas, NumPy, FastAPI, Streamlit, Uvicorn, Matplotlib, Seaborn, Pytest, Pydantic |
| **Setup Notes** | `pip install -r requirements.txt`; run `python -m src.train` to train models and generate artifacts. |
| **Assumptions & Limitations** | 1. **Decision support only:** Advisory outputs; human fleet dispatchers retain final operational authority. <br>2. **Cold-start routes:** New routes without 7d/14d history fall back to route-type medians with wider confidence intervals. <br>3. **Acute disruptions:** Sudden flash closures or transit strikes occurring $<24$ hours out cannot be detected by D-1 booking signals. |

---

### Section 7: AI and Attribution
| Form Field | Entry |
| :--- | :--- |
| **AI Tools Used** | Antigravity AI Coding Assistant |
| **Purpose of Use** | Accelerated code generation, boilerplate test scaffolding, and report drafting. All business logic, chronological splitting, feature engineering, and model validation were designed and verified by the candidate. |
| **Attribution** | Scikit-learn, LightGBM, FastAPI, Streamlit, Seaborn, Pandas open-source packages. Dataset generated synthetically under CC-BY-4.0 license. |

---

### Section 8: Declarations
| Declaration | Status |
| :--- | :---: |
| **Originality** | Confirmed (All code written for this specific assessment). |
| **Ability to Explain** | Confirmed (Ready for the 15-minute walkthrough and 10-minute live debugging task). |
| **No Sensitive Data / Secrets** | Confirmed (Zero PII; `.env.example` placeholder only; no credentials). |
| **Links Verification** | Confirmed (All relative paths and repository links tested). |

---

### Role-Specific Form Fields:
- **Dataset Source / License:** Synthetic Transit Ridership Dataset (17,425 records across 6 routes, 14 months) generated via `src/data_generator.py` under CC-BY-4.0.
- **Notebook / Report Link:** `notebooks/exploratory_data_analysis.ipynb` & `reports/TECHNICAL_REPORT.md`
- **Model Artifact Location:** `artifacts/best_model.joblib` (503 KB), `artifacts/preprocessor.joblib` (3.5 KB), `artifacts/baseline_model.joblib` (1.2 KB)
- **Evaluation Metrics (Held-out Test):**
  - **LightGBM Champion:** WAPE = **8.89%**, MAE = **2.634 pax**, RMSE = **3.458**, $R^2$ = **0.9107**
  - **Historical Baseline:** WAPE = **27.64%**, MAE = **8.188 pax**, RMSE = **10.051**, $R^2$ = **0.2450**
  - **Improvement:** 67.8% relative WAPE reduction, 5.55 passenger MAE reduction.
- **Inference API / App URL:**
  - FastAPI: `http://localhost:8000/docs` (`app/api.py`)
  - Streamlit: `http://localhost:8501` (`app/dashboard.py`)
