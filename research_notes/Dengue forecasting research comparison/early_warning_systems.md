# Dengue Forecasting and Early Warning Systems Using Statistical (Classical) Methods: Designs and Reported Accuracy

Scope note: this covers statistical and classical methods: Bayesian hierarchical/spatiotemporal GLMMs, (S)ARIMA, LASSO/penalised regression, DLNM+INLA and endemic-channel alarms. ML/deep-learning studies are mentioned only for context. Unless stated otherwise, numbers were taken from the primary paper's full text or abstract, retrieved through Europe PMC.

## Brazil: InfoDengue and Lowe et al. 2014 World Cup early warning (Lancet Infect Dis 2014; eLife 2016)

### Takeaway
The Lowe et al. Brazil system was a spatiotemporal hierarchical Bayesian model driven by seasonal climate forecasts and recent dengue data. It forecast June dengue-risk categories for 553 microregions 3 months ahead. In the real prospective test (June 2014) it had a 57% hit rate for high risk, against 33% for a seasonal-average null model, but more false alarms (23% vs 13%). It got the risk category right in 7 of 12 World Cup host cities. InfoDengue is an operational nowcasting and alert system (delay-corrected cases, Rt, climate and tweets mapped to 4 alert levels). Its published accuracy evidence is a 2016 preprint covering Rio de Janeiro only.

### Cited Findings
- **Lowe et al. 2014, Lancet Infect Dis 14(7):619-626, "Dengue outlook for the World Cup in Brazil: an early warning model framework driven by real-time seasonal climate forecasts"** (DOI 10.1016/S1473-3099(14)70781-9). Probabilistic forecast of dengue risk for all 553 microregions, using real-time seasonal climate forecasts (ECMWF, Met Office, Météo-France, CPTEC) plus the dengue situation observed at issue date. The model is a spatiotemporal hierarchical Bayesian framework giving warnings **3 months ahead**. Trigger thresholds were tuned on past performance for **June 2000-2013**. The forecast said low risk for Brasília, Cuiabá, Curitiba, Porto Alegre and São Paulo; medium for Rio, Belo Horizonte, Salvador and Manaus; and high for Recife (p(high)=19%), Fortaleza (46%) and Natal (48%) — [Europe PMC abstract](https://europepmc.org/article/MED/24841859) / [DOI](https://doi.org/10.1016/S1473-3099(14)70781-9)
- **Lowe et al. 2016, eLife 5:e11285, "Evaluating probabilistic dengue risk forecasts from a prototype early warning system for Brazil"**. This is the out-of-sample evaluation of the forecast issued in March for June 2014. The high-risk event was DIR > 300 cases/100,000. Contingency table, forecast model vs null model (microregion mean DIR for June 2000-2013) — [PMC4775211](https://pmc.ncbi.nlm.nih.gov/articles/PMC4775211/) / [DOI 10.7554/eLife.11285](https://doi.org/10.7554/eLife.11285):
  - Hits 81 vs 46; false alarms 94 vs 55; misses 60 vs 95; correct rejections 318 vs 357.
  - **Hit rate 57% vs 33%; false alarm rate 23% vs 13%; miss rate 43% vs 67%.**
  - When low risk was observed, the model correctly assigned a low warning in **67%** of microregions.
  - Warning rules: low warning if p(low) > 68%; medium if p(low) ≤ 68% (i.e., medium trigger 32%); upgraded to high if p(high) > 18%. Benchmark climatology was p_L=68%, p_M=16%, p_H=16%.
  - Correct category in **7 of 12 host cities**: Fortaleza and Natal (high); Belo Horizonte, Manaus and Salvador (medium); Curitiba and Porto Alegre (low). Misses: Brasília was forecast low (p_H 7%) but observed high (DIR 725); Recife forecast high, observed medium; São Paulo and Cuiabá forecast low, observed medium; Rio forecast medium, observed low.
  - In leave-one-year-out cross-validation for 2000-2013, the model's hit rate beat the null in **every year except 2004**, the lowest-incidence year.
- **InfoDengue (Codeço et al., bioRxiv 2016, "InfoDengue: a nowcasting system for the surveillance of dengue fever transmission")**. Started in Rio de Janeiro as an academia-municipality partnership. Each epidemiological week it pulls climate series, case notifications and social-network (Twitter) activity, applies a probabilistic correction for notification delay, and estimates the effective reproductive number. It then codes transmission into **four alert levels** for each of Rio's 10 health districts. Evaluation used historical data from 2011-2014 and prospective data from Jan-Dec 2015. The preprint claims a "good degree of confidence and accuracy" but the abstract gives no headline metric — [bioRxiv DOI 10.1101/046193](https://doi.org/10.1101/046193); project site [info.dengue.mat.br](https://info.dengue.mat.br)

### Inferences
- The Brazil 2014 result is one of the few truly prospective, pre-registered-style dengue forecast evaluations. The headline is modest: roughly 6 in 10 high-risk microregions caught 3 months ahead, at the cost of about 1 in 4 non-epidemic microregions getting a false alarm. Even so, it clearly beat climatology on hits.
- InfoDengue works mainly as a nowcast/alert tool (situational awareness: delay correction plus Rt thresholds), not a multi-month forecaster. Its accuracy has not been reported in the same hit-rate format.

### Gaps
- I did not retrieve a peer-reviewed InfoDengue paper with quantitative alert accuracy (sensitivity/PPV of alert levels). Later InfoDengue publications exist (e.g., expansion to Brazilian states and Infodengue-Mosqlimate forecasting sprints) but were not checked.

## Singapore: Shi et al. 2016 (LASSO, up to 3 months ahead) and NEA operational use

### Takeaway
Singapore's NEA used a set of LASSO-penalised linear regressions, one per forecast horizon of 1-12 weeks. Inputs were lagged cases, the breeding percentage, temperature, humidity and population. Out-of-sample MAPE rose only from 17% at 1 week to 24% at 3 months, against 29% at 3 months for SARIMA and step-down regression. The system was used operationally and predicted the 2013 epidemic peak (863 forecast vs 842 observed cases/week) months in advance.

### Cited Findings
- **Shi Y, Liu X, Kok SY, Rajarethinam J, Liang S, Yap G, Chong CS, Lee KS, Tan SS, Chin CK, Lo A, Kong W, Ng LC, Cook AR. 2016, Environ Health Perspect 124(9):1369-1375, "Three-Month Real-Time Dengue Forecast Models: An Early Warning System for Outbreak Alerts and Policy Decision Support in Singapore"** — [PMC5010413](https://pmc.ncbi.nlm.nih.gov/articles/PMC5010413/) / [DOI 10.1289/ehp.1509981](https://doi.org/10.1289/ehp.1509981)
  - Method: a separate multivariable linear regression submodel for each forecast window (1-12 weeks), with predictors selected by LASSO. Inputs were weekly dengue cases, NEA's "breeding percentage" (the share of Ae. aegypti among Aedes breeding sites), temperature, humidity and population, all at multiple lags.
  - Design: trained on 2001-2010, validated on **2011-2012**, then applied to the 2013 epidemic (22,170 cases, the largest on record). A separate leave-one-year-out exercise was also run.
  - Accuracy: **MAPE 17% (95% CI 16-19%) at 1 week, rising to 24% (22-26%) at 3 months (12 weeks)**. SARIMA and step-down linear regression both had **MAPE 29% at 3 months**. LASSO matched the alternatives in the first 2-week windows and beat them at longer horizons. LASSO and SARIMA 95% prediction intervals were over-covered (conservative), while step-down intervals sometimes fell below nominal coverage.
  - Operational: "we describe a new approach to forecasting dengue that is used by Singapore's NEA in planning vector control and in public communication." The authors say the system predicted the large outbreaks of **both 2013 and 2014 over 10 weeks in advance**. In late March 2013 it forecast an earlier-than-usual rise peaking at about 800/week in June: the forecast peak was **863 cases in week 26** vs an observed **842 cases/week in week 25**. The forecasts were used for hospital bed management, pre-emptive source reduction and staff recruitment, and to launch the national Dengue Campaign in April, 2 months earlier than its usual June start.
  - Caveat noted by authors: early in the 2013 epidemic the forecast was a mild rise that the actual epidemic exceeded, and overall the model predicted a slightly faster end and smaller size than occurred.

### Inferences
- This is a penalised-regression statistical approach (often labelled "machine learning" by the authors), with a clear out-of-sample validation and documented operational use. It is a strong example of a horizon-specific direct forecasting design.

### Gaps
- Current NEA operational practice after 2016 (whether the LASSO system is still in use or has been replaced) was not verified.

## Vietnam (D-MOSS), Lowe et al. Ecuador, and Colón-González et al. 2021 superensemble

### Takeaway
The D-MOSS Vietnam system forecasts monthly province-level dengue incidence 1-6 months ahead for all 63 provinces. It uses a superensemble of Bayesian spatiotemporal models fed by Earth observations, GloSea5 seasonal forecasts and lagged cases. It was only slightly better than a constant-incidence baseline on CRPS (66.8 vs 79.4 at 1-3 month lead) but detected considerably more outbreaks (69% vs 54.5%). In Ecuador (Machala 2016), a Bayesian model driven by El Niño and climate forecasts correctly predicted an early March peak when issued in January.

### Cited Findings
- **Colón-González FJ, Soares Bastos L, Hofmann B, ... Brady OJ, Lowe R. 2021, PLoS Med 18(3):e1003542, "Probabilistic seasonal dengue forecasting in Vietnam: A modelling study using superensembles"** — [PMC7971894](https://pmc.ncbi.nlm.nih.gov/articles/PMC7971894/) / [DOI 10.1371/journal.pmed.1003542](https://doi.org/10.1371/journal.pmed.1003542)
  - Target: monthly dengue incidence per province (63 provinces), **1 to 6 months ahead**, with forecast timing aligned to Vietnamese planning deadlines. Bayesian spatiotemporal models were fit to 19 years (2002-2020) of province data and combined into a superensemble. Inputs were Earth observations (temperature, precipitation, humidity, wind), UK Met Office GloSea5 seasonal forecasts, population, land cover and lagged dengue cases.
  - Evaluation: according to the full text (via WebFetch summary), expanding-window time-series cross-validation over Jan 2007-Dec 2016, plus a prospective forecast for May-Oct 2020.
  - Accuracy at 1-3 month lead (historical): **superensemble CRPS 66.8 (95% CI 60.6-148.0) vs baseline 79.4 (78.5-80.5)**. The baseline forecasts the same incidence rate every month. The paper calls this "slightly more accurate… albeit with larger uncertainty".
  - **Outbreak detection 69% (superensemble) vs 54.5% (baseline).**
  - Prospective May-Oct 2020: **CRPS 110 (102-575) vs 125 (120-168) for baseline**.
  - Predictions were most accurate in southern Vietnam (semi-regular seasonal transmission). The authors say "the majority of past outbreaks could be predicted up to 3 months ahead" and that skill declined beyond 3 months. The superensemble beat each of its component models across horizons and settings.
  - The full-text summary from WebFetch also reported "relative value" (economic value vs no forecast) in about 76% of provinces at 1-month lead. Treat this as unverified; it is not in the abstract.
- **Lowe R, Stewart-Ibarra AM, Petrova D, et al. 2017, Lancet Planet Health 1(4):e142-e151, "Climate services for health: predicting the evolution of the 2016 dengue season in Machala, Ecuador"**. Bayesian hierarchical mixed model using precipitation, minimum temperature and Niño3.4 forecasts. It was initiated on 1 Jan 2016 and produced monthly forecasts to Nov 2016, then evaluated retrospectively against observed data. It **correctly forecast an early peak in March 2016, with a 90% chance of exceeding the mean incidence of the previous 5 years**. Correcting for chikungunya cases misrecorded as dengue in 2015 improved magnitude prediction — [DOI 10.1016/S2542-5196(17)30064-5](https://doi.org/10.1016/S2542-5196(17)30064-5)

### Inferences
- Across the Lowe-group systems (Brazil, Ecuador, Vietnam), the main gain over climatology/constant baselines lies in outbreak detection and hit rate (roughly +15 to +24 percentage points). Gains in overall probabilistic error (CRPS) are modest. Skill fades beyond about 3 months.

### Gaps
- The Ecuador paper gave a single-season evaluation with no hit-rate table in the abstract. No multi-season out-of-sample statistics were retrieved.
- I did not find published results from the planned 2-year prospective evaluation of D-MOSS in Vietnam.

## WHO/TDR EWARS / EWARS-csd (Hussain-Alkhateeb, Kroeger et al.)

### Takeaway
EWARS began as an endemic-channel and alarm-indicator toolkit; its current form, EWARS-csd, uses DLNM + INLA Bayesian models in R. It predicts district-level dengue outbreaks (hospitalisations exceeding an endemic-channel threshold, mean + z·SD) from climate, entomological and epidemiological alarm indicators, with lags of several weeks. Retrospective validations report high sensitivity (often 80-100%) and moderate PPV (40-88%). These are calibration-optimised numbers, so they are optimistic. EWARS-csd is reported as implemented in 17 countries.

### Cited Findings
- **Hussain-Alkhateeb L, Kroeger A, Olliaro P, Rocklöv J, Sewe MO, et al. 2018, PLoS ONE 13(5):e0196811, "Early warning and response system (EWARS) for dengue outbreaks: Recent advancements towards widespread applications in critical settings"**. Field testing in **30 health districts in Brazil, Malaysia and Mexico**. The outbreak indicator is weekly hospitalised dengue cases; alarm indicators include temperature, rainfall, humidity, probable cases, mean patient age, ovitraps and serotype change. The second-generation EWARS-R improved on the first generation in all three countries. **Sensitivity 83-99% (Brazil), 50-99% (Malaysia), 79-100% (Mexico); PPV 40-88% (Brazil), 71-80% (Malaysia), 50-83% (Mexico).** Examples: Brazil probable cases sens 99%/PPV 59%, rainfall 88/88, multiple indicators 83/40. Mexico probable cases 100/83, multiple 84/77. Malaysia temperature 99/80, serotype change (2-week lag) 50/71; ovitrap (3-week lag) 79/60 — [PMC5935393](https://pmc.ncbi.nlm.nih.gov/articles/PMC5935393/) / [DOI 10.1371/journal.pone.0196811](https://doi.org/10.1371/journal.pone.0196811)
- **Schlesinger M, Prieto Alvarado FE, Borbón Ramos ME, Sewe MO, Merle CS, Kroeger A, Hussain-Alkhateeb L. 2024, Front Public Health 12:1323618, "Enabling countries to manage outbreaks: statistical, operational, and contextual analysis of EWARS-csd for dengue outbreaks"**. Retrospective validation in **11 municipalities in 2 Colombian states, 2015-2020**. Alarm indicators were mean temperature and rain sum; the outbreak indicator was weekly hospitalisations. Model: DLNM + INLA; lags of **8-14 weeks** tested; calibration over run-in year, z-value and lag. **Median of the optimal calibration per municipality: sensitivity 97%, specificity 94%, PPV 75%, NPV 99%.** It was weaker in the highly endemic municipality at predicting non-outbreak weeks. The earlier EWARS version had failed in Colombia because of seasonal inconsistency and low-endemicity settings. "EWARS-csd is being implemented in 17 countries." — [PMC10834665](https://pmc.ncbi.nlm.nih.gov/articles/PMC10834665/) / [DOI 10.3389/fpubh.2024.1323618](https://doi.org/10.3389/fpubh.2024.1323618)
- **Cardenas R, Hussain-Alkhateeb L, Benitez-Valladares D, Sánchez-Tejeda G, Kroeger A. 2022, BMC Infect Dis 22:235, "The Early Warning and Response System (EWARS-TDR) for dengue outbreaks: can it also be applied to chikungunya and Zika outbreak warning?"** Covers 9 districts in Mexico and 1 city in Colombia (Cúcuta). Dengue 2012-2016, split into a run-in period and an analysis period. **Mexico dengue: sensitivity 100%, PPV 83%. Colombia dengue: sensitivity 92%, PPV 68%.** Zika alarm-to-outbreak lead time was 4-5 weeks — [PMC8902764](https://pmc.ncbi.nlm.nih.gov/articles/PMC8902764/) / [DOI 10.1186/s12879-022-07197-6](https://doi.org/10.1186/s12879-022-07197-6). Discrepancy: the 2021 preprint version reported Mexico dengue sensitivity 92%, PPV 68% and a dengue lead time of 6-8 weeks — [Research Square 10.21203/rs.3.rs-498936/v1](https://doi.org/10.21203/rs.3.rs-498936/v1)
- **Benitez-Valladares D, Kroeger A, Tejeda GS, Hussain-Alkhateeb L. 2021, PLoS NTD 15(12):e0009261**. In 2017, **20 Mexican health districts** ran EWARS operationally. Of 11 analysed, 9 had alarms with no subsequent outbreak and 2 had alarms followed by outbreaks. Vector-control coverage was significantly higher in the non-outbreak districts, and in outbreak districts response began late and at low intensity. The authors argue some "false alarms" may be averted outbreaks — [PMC8717980](https://pmc.ncbi.nlm.nih.gov/articles/PMC8717980/) / [DOI 10.1371/journal.pntd.0009261](https://doi.org/10.1371/journal.pntd.0009261)

### Inferences
- EWARS metrics are retrospective and calibrated per district: optimal lag, z-value and run-in are chosen to maximise sensitivity and PPV. They are closer to in-sample performance than to true prospective skill and are not directly comparable with the Brazil or Thailand out-of-sample numbers.
- Outbreaks are defined relative to an endemic channel (mean + z·SD of historical hospitalisations), so "outbreak" is a local, relative concept.

### Gaps
- No peer-reviewed EWARS-csd dengue validation with metrics was found for Sri Lanka or Nepal in this search, although Sri Lanka collaborators (Liyanage) work with the same Umeå/Rocklöv group. The list of 17 implementing countries was not retrieved.

## 2015 US government Dengue Forecasting Project (Iquitos, San Juan) and Johansson et al. 2019 PNAS

### Takeaway
Sixteen teams forecast peak week, peak incidence and season total for 4 out-of-sample test seasons (2009/10-2012/13) in Iquitos and San Juan. Skill varied widely, was low early in the season, and was lowest in high-incidence seasons. Statistical models generally beat mechanistic ones. A simple SARIMA baseline had the best calibration and the top peak-week skill. The multi-team ensemble was the only forecast to beat the null for every target.

### Cited Findings
- **Johansson MA, Apfeldorf KM, Dobson S, et al. 2019, PNAS 116(48):24268-24274, "An open challenge to advance probabilistic forecasting for dengue epidemics"** — [PMC6883829](https://pmc.ncbi.nlm.nih.gov/articles/PMC6883829/) / [DOI 10.1073/pnas.1909865116](https://doi.org/10.1073/pnas.1909865116)
  - **16 teams**. Targets: **peak incidence, peak week and total season incidence**, as binned probabilistic forecasts made at **4-week intervals** (weeks 0-24 of the season). Data covered 13 seasons for Iquitos and 23 for San Juan.
  - Design: 4 training seasons (2005/06-2008/09). After submitting those, teams got 2 weeks to submit out-of-sample forecasts for **4 testing seasons (2009/10-2012/13)**, i.e., **8 location-seasons** across both cities.
  - Scoring: logarithmic score. Comparators were a null model (equal probability for all outcomes), a **SARIMA baseline** (4 parameters for Iquitos, 5 for San Juan), and a simple equal-weight ensemble of the 16 teams plus baseline.
  - Inputs: all teams used dengue data; 10 of 16 (63%) used climate data; 2 used serotype data.
  - Results: "Many teams outscored both the null model and, for each target except peak week, the baseline model." "The ensemble forecast outperformed most individual models and was the only forecast to outperform the null model for every target." The ensemble beat the null for all targets in all 8 seasons. The top teams differed for every target except peak week in Iquitos.
  - Modelling approach: log scores were higher for ensemble approaches and lower for models with mechanistic components. "Statistical models generally outperformed mechanistic models." SARIMA "generally performed well compared with more complex models, including having the best overall calibration and the highest skill forecasts for peak week."
  - Skill rose as each season progressed and was lower for seasons with later and higher peaks.
- Project portal: the NOAA/CDC/DoD/OSTP "Dengue Forecasting Project" (2015) on [predict.cdc.gov / dengueforecasting.noaa.gov](https://dengueforecasting.noaa.gov/) (not fetched).

### Inferences
- The challenge is the clearest evidence that well-specified simple statistical models (SARIMA) and ensembles match or beat complex mechanistic and climate-driven models for dengue, especially for peak timing.

### Gaps
- Per-team rankings and exact score differences (SI Table S2) were not extracted.

## Thailand: Lauer et al. 2018 PNAS (province-level annual DHF forecasts)

### Takeaway
A parsimonious regression using preseason (Jan-Mar) incidence, issued each April, forecast annual DHF incidence for 76 provinces. Prospectively over 2010-2014 it beat a 10-year-median baseline in 65% of province-years, cut MAE by 19% (rMAE 81%), and ranked outbreak risk with AUC 0.84. Adding climate covariates did not consistently help.

### Cited Findings
- **Lauer SA, Sakrejda K, Ray EL, Keegan LT, Bi Q, Suangtho P, Hinjoy S, Iamsirithaworn S, Suthachana S, Laosiritaworn Y, Cummings DAT, Lessler J, Reich NG. 2018, PNAS 115(10):E2175-E2182, "Prospective forecasts of annual dengue hemorrhagic fever incidence in Thailand, 2010-2014"** — [PMC5877997](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877997/) / [DOI 10.1073/pnas.1714457115](https://doi.org/10.1073/pnas.1714457115)
  - Target: annual DHF incidence per province (76 provinces), forecast in April from data available then. 87% of cases occur April-December.
  - Training: leave-one-year-out CV over 760 province-years (2000-2009), searching 202 candidate models. The best CV-MAE model (preseason incidence, January rainfall, January temperature, low-season temperature, population) was compared with a sparser incidence-only model. Testing: **380 province-years, 2010-2014, out of sample.**
  - The incidence-only model beat the 10-year-median baseline in **246/380 (65%)** province-years and had **rMAE 81%** vs baseline (a 19% MAE reduction). It also beat the weather+incidence+population (WIP) model (rMAE 93%; better in 57% of province-years). It was the best model in 10 of 13 regions (77%), WIP in 2 (15%) and baseline in 1 (8%). Its 80% prediction intervals covered 80% of observations, vs 70% for WIP.
  - Outbreaks occurred in 38/380 (10%) test province-years. **AUC 0.84** for ordering outbreak probability.
  - Climate covariates "did not consistently add value", although WIP helped in regions with colder or rainier low seasons.

### Inferences
- Lagged-incidence-only models are hard to beat at annual, province scale. This agrees with the Johansson challenge finding that simple statistical models perform well.

### Gaps
- None critical for this question.

## South Asia: Nepal, Bangladesh, Sri Lanka, India, Pakistan

### Takeaway
South Asia has few rigorously validated statistical dengue forecasting or early-warning systems with reported prospective accuracy. Nepal's EWARS is a hospital-based reporting system, not a predictive model, and the 2022 epidemic exposed the lack of alarm thresholds. Recent Bangladesh and Nepal studies mostly compare (S)ARIMA(X) with ML on national monthly data. Sri Lanka has DLNM-based threshold work (Kalutara) pointing toward a 2-month lead-time EWS, with only low-to-moderate predictive accuracy.

### Cited Findings
- **Nepal EWARS (Early Warning and Reporting System)**: established in 1996 under the Epidemiology and Disease Control Division (EDCD) as a hospital-based sentinel system for vector-borne and outbreak-prone diseases. Per the NHRC review, it now runs in 118 hospitals covering all 77 districts. It is reactive case reporting, not forecasting — [NHRC review of surveillance and climate change in Nepal (PDF)](https://nhrc.gov.np/wp-content/uploads/2021/02/Final-review-climate-change-and-survailalcne-in-Nepal.pdf)
- A commentary on Nepal's 2022 epidemic argued that, without predefined alarm-signal criteria, EDCD delayed declaring an epidemic, and called for an efficient early-warning system — [Frontiers in Tropical Diseases 2023, DOI 10.3389/fitd.2023.1217939](https://www.frontiersin.org/journals/tropical-diseases/articles/10.3389/fitd.2023.1217939/full)
- Nepal 2022 epidemic scale: 53,951 cases and 62 deaths by November 2022, vs 17,999 cases in 2019 — [Jha et al. 2025, Trop Med Health, PMC12683827](https://pmc.ncbi.nlm.nih.gov/articles/PMC12683827/)
- **Western Nepal 2026 (BMC Public Health), "Use of time-series forecasting of dengue incidence based on meteorological predictors in Western Nepal"**: per the search snippet, SARIMAX with meteorological variables gave the best out-of-sample forecasting performance. I could not open the full text (paywall/redirect), so its metrics are unverified — [DOI 10.1186/s12889-026-29291-z](https://link.springer.com/article/10.1186/s12889-026-29291-z)
- **Khaki JJ, Acharya BK, Morita K, Pandey BD, Giorgi E. 2025 (medRxiv), "A regression-based approach to modelling recurrent outbreaks: dengue fever in Nepal (2006-2022)"**. Negative binomial regression with "outbreak intensity functions" on annual counts for 77 districts. Descriptive, not a validated forecaster; the authors only outline an extension to outbreak detection — [DOI 10.1101/2025.05.21.25327909](https://doi.org/10.1101/2025.05.21.25327909)
- Other Nepal modelling (MaxEnt climate suitability, Acharya et al. 2018 [PMC5857046](https://pmc.ncbi.nlm.nih.gov/articles/PMC5857046/); compartmental/fractional ODE models fitted by least squares, e.g., Gholami et al. 2025 Sci Rep [PMC11931001](https://pmc.ncbi.nlm.nih.gov/articles/PMC11931001/)) does not report out-of-sample forecast accuracy.
- **Bangladesh**: Chowdhury 2026 (Health Sci Rep) compared ARIMA vs XGBoost on DGHS monthly national data (2000-2023, 565,890 cases; 321,179 in 2023) using RMSE/MAE/MASE — [PMC13097490](https://pmc.ncbi.nlm.nih.gov/articles/PMC13097490/). Hossain et al. 2025 (medRxiv) compared Bayesian SARIMA-Poisson/NB vs SARIMA-ML hybrids on Jan 2000-Aug 2025 hospital data with CRPS, coverage, sensitivity and specificity — [DOI 10.1101/2025.09.14.25335716](https://doi.org/10.1101/2025.09.14.25335716). Faruk 2026 compared NB regression, SARIMAX, XGBoost and LSTM on 2010-2024 monthly data with out-of-sample MAE — [PMC13087637](https://pmc.ncbi.nlm.nih.gov/articles/PMC13087637/). Exact metrics are not in the retrieved abstracts.
- **Sri Lanka**: Liyanage et al. 2022 (Parasites & Vectors) used DLNM plus two-stage hierarchical meta-analysis in Kalutara district, 2010-2019. Larval indices were associated with dengue risk at **1-2 month lags**, with risk thresholds of **BI 2, PI 15, CI 45**. All three indices showed only "low to moderate accuracy" in predicting risk. The authors propose a threshold-based local EWS with a **2-month lead time** — [PMC9351248](https://pmc.ncbi.nlm.nih.gov/articles/PMC9351248/). An earlier Kalutara study modelled ENSO and weather lags — [PMC5129297](https://pmc.ncbi.nlm.nih.gov/articles/PMC5129297/).
- An updated narrative review of dengue forecasting and early warning systems exists for broader context — [PMC12810927](https://pmc.ncbi.nlm.nih.gov/articles/PMC12810927)

### Inferences
- For Nepal specifically, I found no peer-reviewed operational dengue forecast with a prospective hit rate. The field is at the stage of retrospective SARIMA(X)/regression fitting. The 2022 epidemic, roughly 3x the 2019 record, is the motivating event.
- Bangladesh studies are numerous but mostly national-level, monthly and retrospective, with ML-heavy comparisons. None was identified as an operational DGHS system with published accuracy.

### Gaps
- India and Pakistan: not searched in depth within the tool budget. No validated operational statistical dengue EWS with reported accuracy was retrieved for them.
- Colombo-specific forecasting studies with accuracy metrics were not retrieved.
- Metrics for the 2026 Western Nepal SARIMAX paper could not be verified.

## Mexico, Puerto Rico, Thailand and others with clear reported accuracy

### Takeaway
Mexico's operational evidence comes through EWARS (sensitivity 79-100%, PPV 50-83%; 20 districts in 2017). Puerto Rico's comes through the San Juan arm of the 2015 challenge. Thailand's comes through Lauer et al. (65% of province-years better than baseline; AUC 0.84).

### Cited Findings
- Mexico EWARS: sensitivity 79-100% and PPV 50-83% across alarm indicators; probable cases 100%/83%; multiple indicators 84%/77% — [PMC5935393](https://pmc.ncbi.nlm.nih.gov/articles/PMC5935393/). Mexico 2012-2016 dengue: sensitivity 100%, PPV 83% — [PMC8902764](https://pmc.ncbi.nlm.nih.gov/articles/PMC8902764/). Operational in 20 districts in 2017 — [PMC8717980](https://pmc.ncbi.nlm.nih.gov/articles/PMC8717980/)
- Puerto Rico (San Juan): 23 seasons of data, 4 test seasons 2009/10-2012/13; SARIMA baseline (5 parameters) gave the best calibration and peak-week skill; the ensemble beat the null for all targets — [PMC6883829](https://pmc.ncbi.nlm.nih.gov/articles/PMC6883829/)
- Colombia EWARS-csd (11 municipalities, 2015-2020): median sensitivity 97%, specificity 94%, PPV 75%, NPV 99% — [PMC10834665](https://pmc.ncbi.nlm.nih.gov/articles/PMC10834665/)
- Malaysia EWARS: sensitivity 50-99%, PPV 71-80% — [PMC5935393](https://pmc.ncbi.nlm.nih.gov/articles/PMC5935393/)

### Inferences
- Comparison table, by evaluation rigor (most to least prospective):
  1. Brazil 2014 (true prospective, 1 season, 553 units): hit rate 57% vs 33% null; false alarm rate 23% vs 13%.
  2. Thailand 2010-2014 (prospective-style, 380 province-years): 65% beat baseline; MAE -19%; AUC 0.84.
  3. Johansson challenge (blind out-of-sample, 8 seasons): high variability; SARIMA and ensembles were best.
  4. Vietnam (cross-validated plus 1 prospective season): CRPS 66.8 vs 79.4; detection 69% vs 54.5%.
  5. Singapore (temporal hold-out 2011-12 plus operational): MAPE 17% at 1 week to 24% at 12 weeks.
  6. EWARS (retrospective, calibration-optimised): sensitivity roughly 80-100%, PPV roughly 40-88%.

### Gaps
- Puerto Rico Department of Health / CDC operational forecasts after 2015 were not reviewed.
