# Machine-Learning Approaches to Dengue Forecasting: Methods, Reported Accuracy, and Evaluation Rigor

Note on method: these notes come from primary papers (PLoS NTD, PNAS, AJE, IJERPH, Sci Rep, Trop Med Health, PLOS Glob Public Health), from abstracts pulled through Europe PMC, and from one 2026 medRxiv preprint (not yet peer reviewed). Several paywalled full texts (AJE, ScienceDirect, Springer) could not be fetched, so some design details are missing. Those are listed under Gaps.

## Q1. What do systematic reviews say about the quality of this literature?

### Takeaway
The reviews agree on four points. ML (tree ensembles, neural networks, SVR) is the most common family in recent dengue forecasting papers. Validation is often weak: about 20% of models report none, and very few report external or truly out-of-sample validation. Baselines and metrics are not standardized. When results are pooled, ML is not clearly better than well-specified statistical time-series models.

### Cited Findings
**Leung et al. 2023 (PLoS NTD), "A systematic review of dengue outbreak prediction models: current scenario and future directions"**
- 64 studies, 99 models. 39.4% of models used ML (neural networks 23.1%, random forest 15.4%, ensembles 10.3%) — [Leung XY, Islam RM, Adhami M, et al. PLoS Negl Trop Dis 2023;17(2):e0010631. doi:10.1371/journal.pntd.0010631](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)
- Validation: 20.2% of models reported no validation. 75.8% used internal validation only. Only 5.2% reported both internal and external validation — [Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)
- A search-engine summary of the paper said "only 3.9% reported out-of-sample validation". I could not confirm this figure in the fetched full text, so treat it as unverified — [Leung et al. 2023 (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9956653/)
- The medRxiv preprint version reported 78 models from 51 studies, a different count from the published 99/64 — [medRxiv preprint 2022.07.06.22277291](https://www.medrxiv.org/content/10.1101/2022.07.06.22277291v1.full)
- Data and reporting criticisms: 94.7% of models took climate data from agency reports, not real-time sources. Only 22.1% used real-time meteorological stations. 70.7% used climate predictors only. Reporting of methods and performance was inadequate. Performance was reported for 84.8% of models, calibration for 58.6%, and discrimination for 47.5% — [Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)
- Recommendations: add non-climate predictors, use real-time primary data, do more external validation, improve reporting, and consider mechanistic models alongside ML — [Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)

**Sylvestre et al. 2022 (PLoS NTD), "Data-driven methods for dengue prediction and surveillance using real-world and Big Data: A systematic review"**
- 119 studies (2000 to Aug 2020). 67% were published 2016–2020. 60% used an ML approach. 39% used at least one novel data stream (search queries, social media, etc.) — [Sylvestre E, Joachim C, Cécilia-Joseph E, et al. PLoS Negl Trop Dis 2022;16(1):e0010056. doi:10.1371/journal.pntd.0010056](https://doi.org/10.1371/journal.pntd.0010056)
- "The two models with the highest performances were Neural Networks and Decision Trees (52%), followed by Support Vector Machine (17%)." These are counts of which model "won" inside each study, not a head-to-head comparison across studies. The most common significant predictors were rainfall (43%), temperature (41%) and humidity (25%). The authors acknowledge possible selection bias because preprints were excluded — [Sylvestre et al. 2022](https://doi.org/10.1371/journal.pntd.0010056)

**Lober, Rodrigues & Roster, "Neural networks for dengue forecasting: a systematic review"** (arXiv 2021, revised Dec 2025)
- 62 papers. Most used shallow feed-forward neural networks with historical incidence plus climate inputs. "Prediction horizons varied greatly, as did the model selection and evaluation approach." — [Lober L, Rodrigues FA, Roster K. arXiv:2106.12905. doi:10.48550/arXiv.2106.12905](https://arxiv.org/abs/2106.12905)
- "63% of all studies do include at least one such model [a comparator] as a baseline, and in those cases about half of the studies report neural networks as the best performing model." Relative performance against tree-based models "varied across study contexts." Deep networks were "relatively unexplored" — [Lober et al.](https://arxiv.org/abs/2106.12905)

**Benjarattanaporn et al. 2026 (medRxiv preprint), "Dengue Forecasting Models: A Systematic Review and Network Meta-Analysis of Relative Performance"** (PROSPERO CRD420251016662; Sheffield)
- 59 comparative studies, 2014–2024, 20 countries. Model categories: time series in 36 studies (28.6%), ML in 48 (38.1%), deep learning in 23 (18.3%), hybrid in 11 (8.7%), ensemble in 8 (6.3%). The most common single models were linear regression (21), ARIMA (20) and LSTM (19) — [Benjarattanaporn P, Adewo D, Sutton A, Lee A, Dodd PJ. medRxiv 2026. doi:10.64898/2026.02.18.26346534](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)
- The network meta-analysis used RMSE, which 47/59 studies (79.66%) reported. MAE was reported by 62.71% and MAPE by 20.34%. Top SUCRA ranks were k-NN (relative effect −2.32; SUCRA 90.3%), VAR (−2.29; 88.4%), Kalman filter (−2.24; 79.1%), ensembles (−2.13; 75.9%), autoregressive neural network (−2.14; 74.1%) and GLM (−2.14; 72.9%). The naive last-observation-carried-forward model ranked worst. The credible intervals of the top models overlap heavily — [Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)
- Conclusion: "traditional statistical models were often comparable in accuracy to machine learning approaches, with advantages in interpretability and data needs." The authors flag inconsistent reporting of train/test splits (a leakage risk), horizons ranging from 1 to 52 weeks with no standard, and the naive model as the only shared baseline. They also note "potential confounding of method with data and also with analyst skill." 66.1% of studies did not name their software. 59% did not discuss data limitations. They call for standard metrics, shared code and hyperparameters, and open benchmark datasets — [Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)

### Inferences
- Combining the reviews: in roughly 3 out of 4 dengue prediction models, the only validation is internal (often a single split or cross-validation), and only about 5% are tested on external data. "Best model" claims inside single papers (the Sylvestre 52%, the Lober "about half") therefore measure publication framing more than true superiority.
- The NMA is the most direct quantitative comparison. ML does not dominate in it: simple VAR, Kalman filter and GLM sit statistically alongside k-NN and neural networks. The main robust finding is that almost everything beats naive persistence on RMSE, which is a low bar.

### Gaps
- I could not confirm the exact share of studies using random (shuffled) splits versus temporal splits in any review. None of the fetched reviews quantified it directly.
- I could not access Baker et al. or any specific "Roster & Rodrigues" review besides Lober/Rodrigues/Roster. A 2021 review in Research, Society and Development ("Predicting dengue cases through Machine Learning and Deep Learning: a systematic review", rsdjournal.org/index.php/rsd/article/view/19347) exists but was not fetched.

## Q2. Specific ML studies: data, evaluation design, and reported numbers

### Takeaway
Headline accuracy varies enormously with evaluation design. Studies with 1-step-ahead forecasts, lagged cases as inputs, short test windows, or within-sample fitting report R² of 0.95 to 0.99 or "97% accuracy". Rolling-origin, multi-week, out-of-sample evaluations report much more modest gains. When proper count-model or statistical baselines are included, ML sometimes loses.

### Cited Findings
**Guo et al. 2017, SVR, China (PLoS NTD)**
- Data: weekly dengue cases, Baidu search indices and climate (temperature, humidity, rainfall), Guangdong, 2011–2014. Candidate models: SVR, step-down linear regression, GBM, negative binomial, LASSO and GAM — [Guo P, Liu T, Zhang Q, et al. PLoS Negl Trop Dis 2017;11(10):e0005973. doi:10.1371/journal.pntd.0005973](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0005973)
- Design: trained through week 41 of 2014 and validated on the last 12 weeks, plus an outbreak-peak window (weeks 35–46). Horizon was 1 week ahead with rolling windows. External validation used monthly 2014 data from 5 provinces at 1 month ahead. SVR had the lowest RMSE "irrespective of city". External R² values were 0.976 (Yunnan), 0.970 (Guangxi), 0.997 (Hunan), 0.981 (Fujian) and 0.985 (Zhejiang). No naive or persistence baseline was described — [Guo et al. 2017](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0005973)
- Critique (inference): the test covers a single year (the 2014 Guangdong outbreak, about 12 weeks). The horizon is 1 step and search-query data are near-contemporaneous. R² of about 0.97–0.997 at that horizon is the kind of figure persistence alone can come close to.

**Carvajal et al. 2018, RF/GBM vs GAM/SARIMAX, Metro Manila (BMC Infect Dis)**
- Data: weekly dengue incidence and weather (flood, precipitation, temperature, SOI, humidity, wind), 2009–2013. Compared GAM, SARIMAX, random forest and gradient boosting, each run on same-period ("MF") and lagged ("LG") meteorology. RF with lagged meteorology was judged best. Relative humidity was the most important variable — [Carvajal TM, Viacrusis KM, Hernandez LFT, et al. BMC Infect Dis 2018;18:183. doi:10.1186/s12879-018-3066-0](https://doi.org/10.1186/s12879-018-3066-0)
- Critique: the abstract frames the work as explaining the "temporal pattern" and does not describe a temporal out-of-sample forecast. The "MF" arm uses same-week weather, which would not be available for a real forecast. Exact metrics and split design could not be retrieved (full text paywalled/redirected).

**Xu et al. 2020, LSTM, 20 Chinese cities (IJERPH)**
- Data: monthly cases plus climate. Horizon: 1 month ahead. LSTM was compared with BPNN, GAM, SVR and GBM. "LSTM model reduced the average RMSE of the predictions by 12.99% to 24.91% and reduced the average RMSE ... in the outbreak period by 15.09% to 26.82%" compared with the other candidates. Transfer learning helped low-incidence cities — [Xu J, Xu K, Li Z, et al. Int J Environ Res Public Health 2020;17(2):453. doi:10.3390/ijerph17020453](https://doi.org/10.3390/ijerph17020453)
- No naive or seasonal-naive baseline is mentioned in the abstract. Split details could not be fetched (MDPI returned 403).

**Mussumeci & Codeço Coelho 2020, LSTM vs RF vs LASSO, 790 Brazilian cities (Spatial and Spatio-temporal Epidemiology)**
- Data: weekly incidence in 790 cities (InfoDengue), using multivariate predictors including series from similar cities to capture spatial structure. "The LSTM recurrent neural network model attained the highest performance in predicting future incidence." The preprint states LSTM had "the smallest predictive errors ... out of sample." — [Mussumeci E, Codeço Coelho F. Spat Spatiotemporal Epidemiol 2020;35:100372. doi:10.1016/j.sste.2020.100372](https://doi.org/10.1016/j.sste.2020.100372); [medRxiv preprint](https://www.medrxiv.org/content/10.1101/2020.01.23.20018556v1.full)
- No classical time-series or naive baseline was included in the comparison: all three models are ML.

**Zhao et al. 2020, RF vs ANN, Colombia (PLoS NTD)**
- Data: weekly cases for 30 departments plus precipitation, land surface temperature, EVI, population, Gini and education. Training ran 2014–2017 and testing used 2018, with leave-one-season-out cross-validation over 5 years. Horizons were 1–12 weeks. ARIMA was the baseline, and results were reported as relative MAE — [Zhao N, Charland K, Carabali M, et al. PLoS Negl Trop Dis 2020;14(9):e0008056. doi:10.1371/journal.pntd.0008056](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008056)
- National pooled RF MAE was 9.32 at 1 week and 24.56 at 12 weeks. RMAE was below 1 compared with ARIMA. The ANN was "slightly worse" than RF. The national pooled RF beat department-specific RF in 28 of 30 departments — [Zhao et al. 2020](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008056)
- This is one of the stronger designs: a temporal holdout, multiple horizons and a statistical baseline. It has only one held-out test year, though.

**Roster, Connaughton & Rodrigues 2022, ML across Brazilian cities (Am J Epidemiol)**
- Data: monthly dengue cases, 2007–2019, Brazilian cities, using epidemiological and meteorological features. Horizon: 1 month ahead. RF trained on monthly dengue cases alone "performed best overall". It produced lower errors than a **seasonal naive baseline**, gradient boosting, a feed-forward neural network and SVR. The median test MAE across cities was 12.2 cases, falling to 11.9 when the algorithm and feature set were chosen per city. "Different models worked best in different cities." — [Roster K, Connaughton C, Rodrigues FA. Am J Epidemiol 2022;191(10):1803–1812. doi:10.1093/aje/kwac090](https://doi.org/10.1093/aje/kwac090)
- Notably, the best model used dengue lags only, so meteorological features did not improve overall performance.

**Nguyen et al. 2022, deep learning, 20 Vietnamese provinces (PLoS NTD)**
- Data: monthly incidence and 12 meteorological variables, 1997–2016. Trained 1997–2013 and tested 2014–2016 (36 months), a temporal holdout. Compared CNN, Transformer, LSTM and attention-LSTM (LSTM-ATT) against Poisson regression, SARIMA, XGBoost, SVR and SVR-linear. The primary horizon was 1 month, with multi-step forecasts up to 3 months and up to 6 months for Hanoi — [Nguyen V-H, Tuyet-Hanh TT, Mulhall J, et al. PLoS Negl Trop Dis 2022;16(6):e0010509. doi:10.1371/journal.pntd.0010509](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232166/)
- LSTM-ATT had average ranks of 1.60 (RMSE) and 1.95 (MAE) across the 11 models, and beat LSTM in 13 (MAE) or 14 (RMSE) of 20 provinces. Hanoi example: LSTM-ATT RMSE 6.630 and MAE 3.457, against LSTM RMSE 7.999 and MAE 4.926. The authors list as limitations the 36-month test window, reporting delays, and missing data imputed from monthly averages — [Nguyen et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232166/)
- No naive or persistence baseline. The Transformer did not win.

**Chen & Moraga 2025, statistical vs ML, Rio de Janeiro (Trop Med Health)**
- Data: weekly InfoDengue cases plus temperature and humidity, 2016–2023. Evaluation used a rolling moving window (312-week training window, re-fit weekly through 2023, a single test year) at horizons of 1, 2, 3, 4, 8 and 12 weeks. Statistical models: AR, MA, ARIMA, ETS, VAR and SARIMAX. ML models: SVM, RF, XGBoost, LSTM and Prophet — [Chen X, Moraga P. Trop Med Health 2025;53:52. doi:10.1186/s41182-025-00723-7](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984044/)
- Results at 1 week: ARIMA without covariates had MAE 78.26 and MAPE 18.25%. LSTM with covariates had MAE 71.35 and MAPE 22.31%, so its MAE was better but its MAPE was worse. An LSTM–SARIMAX ensemble had MAE 65.24 and MAPE 15.82%. At 12 weeks the best model was Prophet with covariates (MAE 342.47, MAPE 61.86%). No explicit naive baseline was reported — [Chen & Moraga 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984044/)
- Implication: at 1 week, the best ML model improves MAE over plain ARIMA by only about 9% (78.26 to 71.35), and the hybrid improves it by about 17%. Error grows by roughly 4–5x by 12 weeks.

**Wu et al. 2025, ensembles of 11 models, 187 locations (PNAS)**
- Data: Brazil, Colombia, Malaysia, Mexico, Thailand, Iquitos and San Juan, with surveillance data plus Google Trends. Components: AR, ARGO, ARGONet, VAR variants, SIR, ETS, a stacked ML model (elastic net and SVM), plus persistence and seasonal baselines. Evaluation used rolling out-of-sample forecasts at 1–3 months, plus a prospective real-time test from May 2022 to July 2023 — [Wu S, Meyer AG, Clemente L, et al. PNAS 2025. doi:10.1073/pnas.2422335122](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)
- Ensemble mean percent absolute error was 38.5%, 54.5% and 62.7% at 1, 2 and 3 months, against 40%, 55% and 64% for the best single model (VAR). Standard component models were "nearly all unable to outperform a naive persistence baseline in terms of percent absolute error averaged across locations." Ensembles beat persistence consistently, and an ensemble was best in 14 of 15 country-horizon combinations — [Wu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)

**Sesay, Ngunyi & Imboga 2026, BiLSTM-NB vs count models, Freetown, Sierra Leone (PLOS Glob Public Health)**
- Evaluation used an expanding-window, rolling-origin design at 1–3 months, with at least 48 months of training and n = 32 aligned test points per horizon. Climate inputs were restricted to lag 1 or earlier "to prevent look-ahead", and no test-era data were used for model selection. **BiLSTM-NB lost**: INGARCH-NB had the best mean log score, and Diebold-Mariano tests favored INGARCH-NB at all horizons (p ≤ 0.034) — [Sesay MM, Ngunyi A, Imboga H. PLOS Glob Public Health 2026. doi:10.1371/journal.pgph.0005404](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029795/)

**Bangladesh studies (2025–2026)**
- Faruk 2026 (Health Sci Rep): monthly data 2010–2024, comparing NB regression, XGBoost, LSTM and SARIMAX with out-of-sample test MAE. "Machine-learning models showed significant test overfitting." **SARIMAX had the lowest test MAE (17.0).** The paper concludes that "robustness of models is superior to the complexity of the algorithms." — [Faruk O. Health Sci Rep 2026. doi:10.1002/hsr2.72207](https://doi.org/10.1002/hsr2.72207)
- Liu, Hossain & Hossain 2025 (Sci Rep): monthly data for only Jan 2022–Dec 2023 (24 months), 5 divisions, with a temporal 70/30 split (about 7 test months). Compared SARIMA, XGBoost, SVR and MLP. XGBoost was best, with Dhaka MAPE 12.9%. **The reported error values are internally inconsistent.** The fetched text says "RMSE of 109, an MAE of 127" while a search snippet says MAE 109.93 and RMSE 127.39. MAE cannot exceed RMSE, so the first version is a transcription or reporting error — [Liu B, Hossain MF, Hossain S. Sci Rep 2025. doi:10.1038/s41598-025-19752-7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12521351/)
- Other Bangladesh ML papers, from search snippets only (designs not verified): RF with R² = 0.95 and RMSE 34.64 for Dhaka; XGBoost R² 0.84 against LightGBM R² 0.095; classification "accuracy" of 97% (SVM), 92% (XGBoost) and 97.5% (ANN) — [search results incl. PMC12063067](https://pmc.ncbi.nlm.nih.gov/articles/PMC12063067/), [PMC12702403](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12702403/), [ResearchGate Dhaka case study](https://www.researchgate.net/publication/370617601_Prediction_of_Dengue_using_Machine_Learning_Algorithms_Case_Study_Dhaka)

**DengAI (DrivenData) competition**
- Task: predict weekly cases in San Juan (Puerto Rico) and Iquitos (Peru) from NOAA/NASA climate data. Scored by MAE on held-out future years. 17,697 people joined — [DrivenData DengAI](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/leaderboard/)
- The official benchmark is a negative binomial regression on climate. Training data: 936 weeks for San Juan and 520 for Iquitos — [DrivenData benchmark blog](https://drivendata.co/blog/dengue-benchmark)
- A practitioner write-up reports that "the top achievers only get to a MAE of 13 and a bit, while the people ranked around position 100 only make it to 17." — [Towards Data Science/Medium, "Beating DrivenData's DengAI competition Benchmark in 5 minutes"](https://medium.com/data-science/beating-datadrivens-dengai-competition-benchmark-in-5-minutes-c12e86c4331c) (secondary; the leaderboard did not render when fetched)
- A Prophet-based entry reported test MAE of 14.64 (San Juan) and 3.45 (Iquitos) — [search result, GitHub/ResearchGate projects](https://www.researchgate.net/publication/339473958_Predicting_Dengue_Spread_in_San_Juan_and_Iquitos_using_Machine_Learning)

### Inferences
- The test inputs are not lagged observed cases (only climate is provided), and the test horizon is multi-year. Under those conditions, the best-tuned ML ensembles in DengAI land in a narrow MAE band (about 13 to 17), and the "beat the benchmark in 5 minutes" framing suggests that simple seasonal approaches are close to the top. The benchmark's own MAE was not verified (I recall about 25.8 but could not confirm it).
- Two studies, Roster 2022 and Zhao 2020, included a seasonal-naive or ARIMA baseline and a temporal holdout, and found ML better. In both, the gains were modest and city- or horizon-dependent.
- In rigorous rolling-origin designs (Wu 2025, Sesay 2026, Faruk 2026), ML or deep models often fail to beat statistical or count models. Even ensembles reduce error only slightly against the best single statistical model (38.5% vs 40% PAE at 1 month).

### Gaps
- Exact per-model RMSE tables for Guo 2017, Carvajal 2018 and Xu 2020 were not retrievable.
- I could not verify the DengAI official top score, the benchmark MAE, or the winner's method from the leaderboard page.
- No Nepal-specific or India-specific ML forecasting study was verified in this pass.

## Q3. Against proper baselines at multi-week horizons, how much does ML win by, and when does it lose?

### Takeaway
With temporal out-of-sample testing and a sensible baseline, ML gains are usually in the single digits to about 25% in RMSE/MAE, and they shrink or reverse as the horizon lengthens. There are documented cases where ML loses to SARIMAX or count models, and where most individual models fail to beat persistence.

### Cited Findings
- Wu et al. 2025 (PNAS): most standard component models could not beat naive persistence on PAE averaged over 187 locations. Only ensembles beat it consistently. Ensemble PAE vs the best single model (VAR) was 38.5 vs 40% (1 month), 54.5 vs 55% (2 months) and 62.7 vs 64% (3 months) — [Wu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)
- Roster et al. 2022 (AJE): RF beat seasonal naive at 1 month. Median city MAE was 12.2, or 11.9 with per-city model selection. The abstract does not give the seasonal-naive MAE — [Roster et al. 2022](https://doi.org/10.1093/aje/kwac090)
- Zhao et al. 2020: RF had RMAE below 1 against ARIMA at 1–12 weeks, while absolute MAE rose from 9.32 to 24.56 — [Zhao et al. 2020](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008056)
- Chen & Moraga 2025: at 1 week, LSTM with covariates cut MAE about 9% against ARIMA (71.35 vs 78.26), but its MAPE was worse (22.31% vs 18.25%). At 12 weeks, MAPE for the best model was 61.86% — [Chen & Moraga 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984044/)
- Xu et al. 2020: LSTM was 12.99–24.91% lower in RMSE at 1 month against other ML/GAM models, with no naive baseline — [Xu et al. 2020](https://doi.org/10.3390/ijerph17020453)
- ML lost in Sesay et al. 2026, where BiLSTM-NB was significantly worse than INGARCH-NB (p ≤ 0.034) at 1–3 months — [Sesay et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029795/)
- ML lost in Faruk 2026 (Bangladesh), where SARIMAX had the best test MAE (17.0) and XGBoost and LSTM overfit — [Faruk 2026](https://doi.org/10.1002/hsr2.72207)
- The NMA found traditional statistical models "often comparable" to ML, with VAR, Kalman filter and GLM statistically similar to k-NN and neural networks — [Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)

### Inferences
- A fair overall summary: ML is competitive, occasionally better by about 10–25% at short horizons, rarely decisively better at 2–3 months or more, and sometimes worse. Ensembles and hybrids (ML plus SARIMAX, or multi-model stacks) are the most consistent winners.

### Gaps
- None of the verified studies reported skill scores against seasonal naive over many test seasons (for example, at least 5 held-out years). Most use 1–3 test years.

## Q4. Reported accuracy metrics, and why high R² or "accuracy %" can mislead

### Takeaway
High R² (0.95 to 0.997) and classification accuracies above 90% mostly arise from three things: 1-step-ahead forecasting with lagged cases as inputs, which persistence alone already exploits; random or within-sample splits; and same-period or future covariates. Rigorous studies report MAE/RMSE/MAPE or log scores under rolling-origin evaluation, and those numbers are far less impressive.

### Cited Findings
- Examples of very high reported fit: Guo 2017 SVR external R² of 0.970–0.997 at 1-week or 1-month horizons in a single outbreak year — [Guo et al. 2017](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0005973). From search snippets (designs unverified), Bangladesh RF R² 0.95 and SVM/ANN "97–97.5% accuracy" — [PMC12702403](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12702403/)
- Same-period meteorology was used as a model input: the Carvajal 2018 "MF" dataset — [Carvajal et al. 2018](https://doi.org/10.1186/s12879-018-3066-0)
- Explicit leakage controls (lag-1-only climate, rolling-origin, no test-era model selection) appear in only a minority of papers. Sesay 2026 is an example. The NMA flags inconsistent reporting of train/test splits as a leakage risk — [Sesay et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029795/); [Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)
- Lack of standard metrics: RMSE 79.66%, MAE 62.71%, MAPE 20.34%, and 6 studies with non-standard metrics in the NMA set. Horizons ranged from 1 to 52 weeks — [Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)
- Leung 2023: 20.2% of models had no validation at all, and only 5.2% had external validation — [Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)
- Reporting errors occur even in high-profile venues. In Liu et al. 2025 (Sci Rep) the reported MAE exceeds RMSE in one version of the numbers, which is mathematically impossible — [Liu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12521351/)

### Inferences
- Dengue series are strongly autocorrelated. At a 1-week or 1-month horizon, any model fed last period's cases will reach a high R², so R² without a persistence or seasonal-naive comparison carries little information about skill.
- "Accuracy %" from classification framings (outbreak vs no outbreak) depends on class balance and threshold. Without balanced metrics and temporal splits, such numbers cannot be compared with regression forecasting results.
- Random k-fold splits on time series let a model see the future (neighboring weeks) during training. This inflates scores, and the dengue reviews say it remains under-reported.

### Gaps
- I found no quantitative review stating the share of dengue ML papers using random rather than temporal splits. No paper in this pass directly re-evaluated a published high-R² model under a temporal design to quantify the inflation.
