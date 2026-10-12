# Simple models keep winning at dengue forecasting

Yes, many research teams forecast dengue cases and outbreaks, and their results look a lot like DengueWatch Nepal's. The strongest studies find the same thing the student found: **a simple statistical model built from past case counts is very hard to beat**. In a 16-team US government dengue challenge, a small seasonal time-series model with 4–5 parameters had the best calibration and the best peak-week forecasts. Models that used climate data or more complex designs did worse on average ([Johansson et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6883829/)). Machine learning (ML) papers often claim 95–99% accuracy. But most of those papers test weakly: about 20% of dengue models report no validation at all, and only about 5% are tested on outside data ([Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)). When ML is tested carefully, it often loses to statistical or count models ([Sesay et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029795/); [Faruk 2026](https://doi.org/10.1002/hsr2.72207)). The only "complex" method that wins reliably is an ensemble, meaning the average of several models ([Wu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)). So LightGBM losing to a simple GLM on about 8 seasons of data is the expected result, not a bug. Simple-guess baselines (persistence, seasonal average, endemic channel) are well documented and widely used. The endemic channel's known weakness, that past outbreak years inflate the threshold, is exactly the problem the student hit with mean + 2 SD. Weather helps forecasts only sometimes, mostly at longer lead times and with local data. A national average across mountains and lowlands is unlikely to help at 4 weeks. The student's results are honest and in line with the careful end of the literature. Where the literature is stronger is in data volume (more seasons, more places), longer test periods, and stronger baselines.

## A few words you will see in this report

| Term | Plain meaning |
|---|---|
| Persistence (naive) forecast | "Next month will look like this week." It copies the latest number forward. |
| Seasonal naive / climatology / historical average | "This week will look like the same week in past years." |
| Endemic channel | A band of "normal" weekly cases built from past years. Crossing the top line means "outbreak". Introduced for spreadsheets by Bortman (1999). |
| SARIMA | A classic time-series model that uses recent values plus the same season last year. |
| Horizon / lead time | How far ahead you forecast (e.g., 4 weeks). |
| Out-of-sample / rolling-origin test | Testing only on data the model never saw, moving forward one season at a time, like real life. |
| Sensitivity (hit rate) | Share of real outbreaks the warning caught. |
| PPV | When the alarm rings, how often an outbreak really follows. |
| MAE / RMSE / MAPE | Ways to measure average forecast error. Smaller is better. |
| R² | How closely forecasts track the real curve (1 = perfect). Easy to inflate. |
| CRPS / log score / WIS | Scores for forecasts that give a range, not just one number. |
| Ensemble | Averaging several models' forecasts. |

## Research teams across four continents forecast dengue much like DengueWatch

Dengue forecasting is an active research field with operational systems in Singapore, Brazil, Vietnam and Latin America. The main designs are close to the student's. They forecast weekly or monthly counts a few weeks to months ahead, or they raise an outbreak alarm when cases are expected to cross a threshold. The table below lists the best-tested statistical systems.

| Study | Place | Method | Horizon | How it was tested | Main result |
|---|---|---|---|---|---|
| Shi et al. 2016 | Singapore (national, weekly) | LASSO regression, one model per horizon; cases, weather, mosquito data | 1–12 weeks | Trained 2001–2010, tested 2011–2012, then used live in 2013 | **MAPE 17% at 1 week, 24% at 12 weeks**, vs 29% for SARIMA at 12 weeks. Forecast 2013 peak of 863/week vs 842 observed ([PMC5010413](https://pmc.ncbi.nlm.nih.gov/articles/PMC5010413/)) |
| Lowe et al. 2014/2016 | Brazil, 553 microregions | Bayesian model with seasonal climate forecasts | 3 months | True prospective test, June 2014 | **Hit rate 57% vs 33%** for a seasonal-average baseline; false alarm rate 23% vs 13% ([PMC4775211](https://pmc.ncbi.nlm.nih.gov/articles/PMC4775211/)) |
| Colón-González et al. 2021 | Vietnam, 63 provinces | Superensemble of Bayesian models plus climate | 1–6 months | Expanding-window test 2007–2016, plus live 2020 | CRPS 66.8 vs 79.4 for a constant baseline ("slightly more accurate"); **outbreak detection 69% vs 54.5%**; skill faded after 3 months ([PMC7971894](https://pmc.ncbi.nlm.nih.gov/articles/PMC7971894/)) |
| Lauer et al. 2018 | Thailand, 76 provinces | Simple regression on early-season cases | Whole year, issued in April | Prospective 2010–2014, 380 province-years | Beat 10-year median in **65%** of province-years; **19% lower MAE**; AUC 0.84. Climate did not consistently help ([PMC5877997](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877997/)) |
| Johansson et al. 2019 | Iquitos (Peru), San Juan (Puerto Rico) | 16 teams, many methods | Season targets, every 4 weeks | Blind test on 4 seasons per city | SARIMA baseline had best calibration and peak-week skill; only the ensemble beat the "equal odds" null on every target ([PMC6883829](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6883829/)) |
| WHO/TDR EWARS-csd (Schlesinger et al. 2024) | Colombia, 11 municipalities | Bayesian lag model (DLNM + INLA) with weather alarms | 8–14 week lags | Retrospective, tuned per town | Median **sensitivity 97%, PPV 75%**. Tuned on the same data, so optimistic ([PMC10834665](https://pmc.ncbi.nlm.nih.gov/articles/PMC10834665/)) |
| EWARS (Hussain-Alkhateeb et al. 2018) | 30 districts in Brazil, Malaysia, Mexico | Alarm indicators vs endemic channel | Weeks | Retrospective field test | Sensitivity 50–100%, PPV 40–88% ([PMC5935393](https://pmc.ncbi.nlm.nih.gov/articles/PMC5935393/)) |

Two patterns stand out. First, the honest prospective numbers are modest. Brazil caught about 6 in 10 high-risk areas, at the cost of a false alarm in about 1 in 4 calm areas. Vietnam's overall error was only slightly better than a flat guess, but it caught more outbreaks. Second, the very high EWARS numbers come from tuning the lag and threshold on the same data used for scoring. That makes them closer to "fit" than to real forecasting skill.

Nepal itself has almost no tested forecasting. Nepal's EWARS is a hospital-based reporting network, now 118 hospitals in all 77 districts. It reports cases; it does not predict them ([NHRC review](https://nhrc.gov.np/wp-content/uploads/2021/02/Final-review-climate-change-and-survailalcne-in-Nepal.pdf)). The 2022 epidemic reached 53,951 cases by November, against 17,999 in 2019 ([Jha et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12683827/)). A commentary argued that Nepal lacked predefined alarm criteria, which delayed the epidemic declaration ([Frontiers 2023](https://www.frontiersin.org/journals/tropical-diseases/articles/10.3389/fitd.2023.1217939/full)). One 2026 study from Western Nepal reports that SARIMAX with weather gave the best forecasts. The research team could only see a search snippet, so its numbers are unverified ([BMC Public Health 2026](https://link.springer.com/article/10.1186/s12889-026-29291-z)). No peer-reviewed operational dengue forecast with a prospective hit rate was found for Nepal. That gap is a real opening for the student's project.

## Simple-guess baselines are well established and very hard to beat

Researchers do propose and test the simple guesses the student used, and forecasting experts insist on them. The standard textbook says every new method should be compared to the mean, naive and seasonal naive methods "to ensure that the new method is better" ([Hyndman & Athanasopoulos, FPP3](https://otexts.com/fpp3/simple-methods.html)). In Mexico, Johansson and colleagues compared 39 models on 28 years of monthly data. The plain "same month in past years" average reached an R² of about 0.24. No weather-only model beat it. Adding recent and seasonal autocorrelation (SARIMA-type models) beat everything at all horizons. The authors said such a model "should always be used as a baseline" ([Johansson et al. 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5036038/)).

Persistence is the main baseline in the big COVID and flu forecast hubs. It is surprisingly strong, as the table shows.

| Study | Baseline | Horizon | How many models beat the baseline |
|---|---|---|---|
| Reich et al. 2019, US flu, 22 models | Historical average | 1 and 4 weeks | 15 of 22 beat it in 6+ of 7 seasons at 1 week; **only 7 of 22 at 4 weeks** ([PMC6386665](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6386665/)) |
| Cramer et al. 2022, US COVID deaths | Persistence | 1–4 weeks | About two-thirds of models; ensemble had 39% less error. **Only the ensemble beat it everywhere** ([PMC9169655](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9169655/)) |
| Lopez et al. 2024, US COVID cases | Persistence | 1–4 weeks | **7 of 22** at state level; only 2 matched or beat it in every state ([PMC11098513](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11098513/)) |
| Bracher et al. 2021, Germany/Poland COVID cases | Persistence | 1–4 weeks | "Almost none" beat it beyond 1–2 weeks for cases ([PMC8397791](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8397791/)) |
| Wu et al. 2025, dengue in 187 locations | Persistence and seasonal | 1–3 months | Standard single models were "nearly all unable" to beat persistence; ensembles beat it consistently ([PMC12377650](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)) |

Persistence is hard to beat at peaks and plateaus, because it predicts a flat line and the curve really is flat there. It does badly when cases rise or fall fast. Models do worst in rising phases, and they overshoot after peaks because they lean on recent values ([Lopez et al. 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11098513/); [Bracher et al. 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8397791/)). This explains the student's "peaks arrive 2–4 weeks late" problem. It is a known weakness of all short-horizon forecasts, not something special to DengueWatch. In dengue, skill was lowest in high-incidence seasons, which are the seasons people care about most ([Johansson et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6883829/)).

### The endemic channel works, but its +2 SD line fails the way the student found

The endemic channel is the standard outbreak alarm across Latin America and in many Asian countries. Bortman (1999) showed how to build it in a spreadsheet from past years ([PAHO IRIS](https://iris.paho.org/handle/10665.2/8562)). Countries use either mean + 2 SD over the past 5 years, or the median with the upper quartile (Q3) as the alarm line ([Badurdeen et al. 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3697990/)). So the student's switch to an upper-quartile channel is a recognised, published option, not a hack.

The student's problem with mean + 2 SD is documented in the literature. Badurdeen and colleagues wrote that "outbreaks in previous years can result in thresholds that are too high; no satisfactory algorithm for recognising past aberrations has been devised" ([Badurdeen et al. 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3697990/)). A 2026 Malaysia study said large outbreaks "inflate historical means and standard deviations" ([Singh et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517397/)). Brady and colleagues tested 102 outbreak definitions on Brazilian data. Different definitions changed the number of outbreaks by up to 10 and the share of outbreak cases by up to 65% over nine years ([Brady et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4429239/)). They also note that countries often require 2 weeks in a row above the line, which is exactly the student's surge rule.

| Threshold study | Result |
|---|---|
| Puerto Rico, +2 SD (cited in Badurdeen 2013) | Only **about 40%** of crossings were followed by a big rise; specificity about 90% ([PMC3697990](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3697990/); [Umaña 2026](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13155432/)) |
| Barbazan et al., +1 SD (cited in Badurdeen 2013) | Sensitivity 66% |
| Malaysia 2017–2023 (Singh et al. 2026) | Classic +2 SD: **sensitivity 0.26, PPV 0.27**. Best log-scale channel: sensitivity 0.60, specificity 0.82, PPV 0.35 ([PMC13517397](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517397/)) |
| Colombia (Umaña et al. 2026) | Shorter look-back windows did better; whether to drop outbreak years depended on transmission level ([PMC13155432](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13155432/)) |
| Flu Moving Epidemic Method, Europe (Vega et al. 2013) | Sensitivity 71.8%, specificity 95.5%, median 1 week timeliness ([Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1750-2659.2012.00422.x)) |

One more lesson matters for the pitch. An endemic channel only says an outbreak is already happening. Health managers often treat crossing the line as a warning, so response starts late; typical response lags are over 2 weeks ([Badurdeen et al. 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3697990/)). A forecast that flags a surge about 3 weeks before it starts targets exactly this gap.

## Machine learning sometimes helps, but the "97% accurate" papers test weakly

ML is now the most common approach in dengue papers. One review of 64 studies found 39% of models used ML ([Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631)); another of 119 studies found 60% did ([Sylvestre et al. 2022](https://doi.org/10.1371/journal.pntd.0010056)). The quality of testing is the problem. In Leung's review, 20.2% of models had no validation, 75.8% had only internal validation, and only 5.2% had external validation. A 2026 network meta-analysis compared 59 studies. It found that "traditional statistical models were often comparable in accuracy to machine learning approaches." The top-ranked groups (k-NN, VAR, Kalman filter, ensembles, GLM) had heavily overlapping uncertainty. Only naive persistence was clearly worst. This review is a preprint and not yet peer reviewed ([Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)).

| Study | Place | ML method | Horizon / test | Baseline? | Result |
|---|---|---|---|---|---|
| Guo et al. 2017 | China | SVR (support vector regression) | 1 week; test ~12 weeks of one outbreak | No naive baseline | External R² 0.970–0.997 ([PLoS NTD](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0005973)) |
| Xu et al. 2020 | 20 Chinese cities | LSTM neural network | 1 month | Other ML only | 13–25% lower RMSE than other models ([IJERPH](https://doi.org/10.3390/ijerph17020453)) |
| Zhao et al. 2020 | Colombia, 30 departments | Random forest, pooled across departments | 1–12 weeks; one test year | ARIMA | RF beat ARIMA; pooled model beat local models in 28 of 30 departments ([PLoS NTD](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008056)) |
| Roster et al. 2022 | Brazilian cities, 2007–2019 | Random forest | 1 month | Seasonal naive | RF on **dengue cases only** won; median MAE 12.2 cases; best model differed by city ([AJE](https://doi.org/10.1093/aje/kwac090)) |
| Nguyen et al. 2022 | Vietnam, 20 provinces | Attention-LSTM | 1–3 months; 36 test months | SARIMA, Poisson | LSTM-ATT ranked best on average; no persistence baseline ([PMC9232166](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232166/)) |
| Chen & Moraga 2025 | Rio de Janeiro | LSTM, XGBoost, RF, Prophet | 1–12 weeks, rolling; one test year | ARIMA | LSTM only ~9% lower MAE than ARIMA at 1 week, with worse MAPE; LSTM+SARIMAX hybrid best ([PMC11984044](https://pmc.ncbi.nlm.nih.gov/articles/PMC11984044/)) |
| Wu et al. 2025 | 187 locations, 5 countries | 11 models incl. stacked ML | 1–3 months, rolling + live | Persistence | Single models mostly lost to persistence; ensemble error 38.5% vs 40% for best single model ([PMC12377650](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)) |
| Sesay et al. 2026 | Freetown, Sierra Leone | BiLSTM neural network | 1–3 months, rolling, leakage-controlled | Count model (INGARCH-NB) | **ML lost** at every horizon (p ≤ 0.034) ([PMC13029795](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029795/)) |
| Faruk 2026 | Bangladesh, 2010–2024 | XGBoost, LSTM | Monthly, out-of-sample | NB regression, SARIMAX | **ML lost**; SARIMAX best (MAE 17.0); ML "showed significant test overfitting" ([Health Sci Rep](https://doi.org/10.1002/hsr2.72207)) |

Why do some papers report 97% accuracy while careful ones show small gains? Three reasons explain most of it. First, many forecast only 1 step ahead and feed last week's cases in as an input. Dengue counts change slowly, so last week already predicts this week well, and R² is high even for a lazy model. Second, some use random splits, so the model trains on weeks right next to the test weeks and peeks at the future. Third, some use same-week weather, which would not be known when making a real forecast, as in one Manila study ([Carvajal et al. 2018](https://doi.org/10.1186/s12879-018-3066-0)). Reporting errors happen too. One 2025 Scientific Reports paper lists an MAE larger than its RMSE in one version, which is mathematically impossible ([Liu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12521351/)).

When ML does win under fair testing, it wins by roughly 10–25% at short horizons, and the gap shrinks at longer horizons. The wins come with lots of data. Zhao pooled 30 departments. Roster used many Brazilian cities over 13 years. Wu used 187 locations. The DengAI public competition gave 936 training weeks for San Juan and 520 for Iquitos ([DrivenData](https://drivendata.co/blog/dengue-benchmark)). The student has about 360 weekly rows for one national series.

## Why LightGBM lost on about 360 weekly rows

The student's LightGBM result matches what both the general forecasting research and the dengue research predict. The evidence points to five reasons.

**The dataset is too small for a tree ensemble.** DengueWatch has about 8 seasons, or roughly 360 weekly rows, from one national series. Only a few of those weeks are surge onsets, so the warning model has very few positive examples to learn from. In the Makridakis study of 1,045 monthly series with 81–126 data points each, ML methods were "dominated across both accuracy measures used and for all forecasting horizons." Even a seasonal naive guess beat half of the ML methods ([Makridakis et al. 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870978/)). In the M4 competition of 100,000 series, all six pure ML methods fell below a simple statistical combination ([Makridakis et al. 2020](https://www.sciencedirect.com/science/article/pii/S0169207019301128)). DengueWatch's series is about the size of the M3 series where ML lost.

**LightGBM wins only with lots of related data.** LightGBM-based methods did win the M5 competition. But M5 had 42,840 related Walmart sales series, plus helpful extra inputs like prices and promotions ([Makridakis et al. 2022](https://www.sciencedirect.com/science/article/pii/S0169207021001874)). That is the setting trees need. DengueWatch has one series. A depth-3, 300-tree model can still memorise the noise of a few seasons. Makridakis found that ML models often fit the training data better but forecast worse: "minimizing the model fit errors would guarantee more accurate post-sample predictions does not hold" ([Makridakis et al. 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870978/)).

**Careful dengue studies show the same pattern.** In Sierra Leone, a neural network lost to a count model at every horizon under a leakage-controlled rolling test ([Sesay et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029795/)). In Bangladesh, XGBoost and LSTM overfit and lost to SARIMAX ([Faruk 2026](https://doi.org/10.1002/hsr2.72207)). Across 187 locations, single models mostly failed to beat persistence; only ensembles did ([Wu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)). In the 2019 challenge, the authors warned about "the potential downside of overly complex forecasting models" ([Johansson et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6883829/)).

**Trees cannot predict numbers bigger than they have seen.** This point is the report's own reasoning, not taken from a cited study. A tree-based model predicts by averaging training examples that look similar. It cannot forecast counts much higher than anything in its training data. Nepal's 2022 epidemic was about three times the 2019 record. In the 2022 test fold, LightGBM had never seen a season that big. A GLM with a log link can extend a growth trend upward; a tree cannot.

**The GLM gets the right structure for free.** The negative binomial GLM is told that cases grow multiplicatively, follow a seasonal wave, and are noisy counts. LightGBM has to learn all of that from a few hundred rows. With little data, built-in structure beats flexibility. This matches the textbook and challenge advice to prefer models with few parameters, such as the 4–5-parameter SARIMA that won peak-week skill in 2019.

A fair reading is that LightGBM did not fail because ML is bad. It lost because this dataset is too small for it. With district-level data pooled across many districts, ML could become competitive, as it did in Colombia ([Zhao et al. 2020](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008056)).

## Weather helps sometimes: at longer leads, with local data, not at 4 weeks nationally

The evidence on weather is mixed, and it splits by how strong the comparison model is. When weather is added to a model that already uses recent cases and seasonality, gains are small. In Mexico, weather gains were "generally very slight," and only 1 of 17 states (Jalisco) showed a clear gain in both test periods. A temperature model chosen for one state "performed poorly" out of sample ([Johansson et al. 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5036038/)). In the 2019 challenge, models using climate data were less well calibrated. The authors noted that models using dengue history alone "may be able to provide equivalent information" ([Johansson et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6883829/)). In Thailand, climate "did not consistently add value" ([Lauer et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877997/)). In Brazil, the best ML model used dengue cases only ([Roster et al. 2022](https://doi.org/10.1093/aje/kwac090)).

Weather looks more helpful against weak baselines, for outbreak or season-level targets, and at longer leads. In Brazil, a climate-driven model beat a seasonal-average baseline on hit rate (57% vs 33%) ([Lowe et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4775211/)). In Barbados, a drought-then-rain model classified outbreak months correctly 86% of the time against 64% for the baseline ([Lowe et al. 2018](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002613)). In Cali, Colombia, climate models lost to autoregressive models at 1 and 3 months but won at 6 months; this comes from a search snippet only ([Eastin et al. 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4155567/)). Vietnam is the counter-example: the climate superensemble was better at 1–3 months but worse than baseline at 6 months ([Colón-González et al. 2021](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1003542)).

The biology explains why weather adds little at 4 weeks. Temperature and rain act on dengue with lags of about 1–3 months, through mosquito breeding and virus incubation. Transmission peaks around 26–29°C and is possible between about 18°C and 34°C ([Mordecai et al. 2017](https://journals.plos.org/plosntds/article?id=10.1371%2Fjournal.pntd.0005568)). By the time you forecast 4 weeks ahead, last month's weather is already showing up in this week's case count. The case count carries the weather signal. Weather can also only explain part of the story; immunity and serotype shifts also drive epidemic years ([Nat Commun 2025](https://www.nature.com/articles/s41467-025-66411-6)).

Nepal adds a special problem: elevation. In Chitwan, minimum temperature 2 months earlier was the strongest predictor, and it pushed cases up by about 44% per °C (IRR 1.437). Rainfall did not stay in the model ([Tuladhar et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417253/)). Lowlands can be suitable for dengue mosquitoes up to 9 months a year, while suitability is spreading fastest at 500–1,500 m ([Acharya et al. 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322031)). Averaging 67 points across mountains and lowlands mixes very different climates. A cold mountain point moves the national average without having anything to do with where cases occur. That likely dilutes the signal. The student's negative minimum-temperature coefficient is the opposite of Chitwan's positive effect, which is a warning sign. The report's own guess is that week-of-year terms already capture the seasonal temperature cycle, so the leftover temperature term picks up noise. The student's 8-week minimum-temperature window does match Chitwan's 2-month lag, so the idea was sensible; the national averaging is the likely weak point.

## How DengueWatch Nepal compares with the literature

| Feature | DengueWatch Nepal | Typical strong studies |
|---|---|---|
| Data length | 8 seasons, ~360 weekly rows, 1 national series | 13–28 years (Johansson); 187 locations (Wu); 30 departments (Zhao) |
| Data quality issues | Network grew ~60 to 118 sites in 2019; no site counts 2020–2023 | Delay correction in InfoDengue; reporting artefacts noted in hubs |
| Targets | 4-week surge probability + 4-week count with p10–p90 | Weekly/monthly counts, peaks, season totals, outbreak categories |
| Models | Logistic regression + negative binomial GLM, ≤8 inputs | SARIMA, Bayesian GLMs, LASSO, ML, ensembles |
| Test design | Rolling origin, 5 test seasons, threshold refit per fold, leakage test | Often 1–4 test seasons; many ML papers use no proper test |
| Baseline | Persistence | Persistence, seasonal average, SARIMA, endemic channel |
| Count result | Beat persistence in 3 of 5 seasons (by 35–54%), lost in 2 (by 17–25%) | Most single models fail to beat persistence at 3–4 weeks |
| Warning result | 4 of 6 onsets caught ~3 weeks early; ≤2 false alarms/season | Brazil 57% hit rate; Vietnam 69% detection; endemic channel +2 SD about 26–40% |
| ML result | LightGBM lost to GLM | ML often loses under careful tests; wins with lots of pooled data |
| Weather | Mixed, coefficients not sensible | Mixed; small gains at short horizons |

Note: the table reads positive percentages against persistence as "lower error than persistence." Check that this matches how the project defined them.

**Where the student is comparable.** The test design is better than most of the dengue literature. Rolling-origin testing over 5 seasons, refitting the threshold in each fold, and a leakage test are what reviews say is missing ([Leung et al. 2023](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010631); [Benjarattanaporn et al. 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full)). Few published dengue ML papers test on 5 or more held-out years. Beating persistence at 4 weeks in 3 of 5 seasons is respectable. Only 7 of 22 COVID case models beat persistence at state level, and almost none did beyond 1–2 weeks in Germany and Poland. Catching 4 of 6 onsets (67%) sits close to Brazil's 57% and Vietnam's 69%, and well above the classic +2 SD channel. The findings that the GLM beat LightGBM and that weather did not reliably help both match the best-designed studies.

**Where the literature is stronger.** Six surge onsets is a very small number. By the report's own calculation, a 95% confidence interval for 4 of 6 runs from roughly 30% to 90%. So the true hit rate could be much lower or higher. Strong studies have more seasons or more places. They also compare against a stronger baseline than persistence, usually SARIMA or a seasonal average. The student has not yet shown that the count model beats a seasonal model, or that the warning beats a plain upper-quartile endemic channel. Strong studies also report scores by epidemic phase, peak-timing error, and range scores (CRPS or WIS) for the forecast interval. Losing to persistence in 2021 and 2025 also matters. In some seasons, the model makes forecasts worse than doing nothing.

## Practical lessons and next steps

The most useful next step is to **add the missing baselines**. These are a seasonal naive (same week last year), a historical average by week, and a simple SARIMA. For the warning model, add a plain upper-quartile endemic channel. Johansson's team says a model with seasonality and autocorrelation "should always be used as a baseline" ([Johansson et al. 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5036038/)). If the GLM beats all of them, the claim is much stronger. Second, **try a simple ensemble**, such as the average of the GLM, persistence and seasonal forecasts. Ensembles are the one robust improvement in every forecasting hub and in the dengue challenge ([Cramer et al. 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9169655/); [Wu et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/)). This may also fix the bad 2021 and 2025 seasons. Third, **if district counts can be extracted from the bulletins, pool the districts** into one model. Pooling helped random forest in 28 of 30 Colombian departments ([Zhao et al. 2020](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008056)). It is also the only realistic route for LightGBM to become useful.

For weather, **use lowland (Terai) or district weather instead of a 67-point national average**, and give weather its best chance at longer leads (2–3 months). Report weather as a tested idea that did not help nationally at 4 weeks. That is an honest, publishable finding. For evaluation, **report results by phase** (rising, peak, falling), the error in peak timing, and how often the p10–p90 range contains the true value. State the uncertainty around "4 of 6." Also adjust for the 2019 jump in sentinel sites, for example with a step indicator or cases per site. Otherwise part of the growth may be the network growing, not dengue.

For the pitch, the honest framing is also the strongest one. A fair one-line summary: "DengueWatch gives Nepal a tested 4-week dengue surge warning. In five seasons of honest back-testing, it caught 4 of 6 surge starts about 3 weeks early, with at most 2 false alarms a season. It used simple, explainable models that beat more complex machine learning, which matches international research." Avoid claiming "accurate peak prediction," since forecasts reach peaks 2–4 weeks late. Avoid "AI-powered" and "weather-driven." Point out that Nepal currently has no predefined alarm thresholds ([Frontiers 2023](https://www.frontiersin.org/journals/tropical-diseases/articles/10.3389/fitd.2023.1217939/full)). The tool's value is turning hindsight into a few weeks of warning. False alarms can be framed carefully too. In Mexico, some "false alarms" were followed by stronger vector control and may have been outbreaks that were prevented ([Benitez-Valladares et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8717980/)).

## Conclusion

The main lesson from this literature is that, in dengue forecasting, the quality of the test matters more than the cleverness of the model. Papers with 97% accuracy and papers where ML loses are often measuring different things. The student's careful testing put DengueWatch on the honest side of that divide. Seen that way, "the simple GLM beat LightGBM and weather did not help" is not a disappointing result. It is the same answer that the US dengue challenge, the M competitions and the latest careful dengue studies reached, now shown for Nepal.

The real frontier for the project is not a fancier model but more and better data. That means more seasons, district-level counts, lowland weather, and stronger baselines to beat. Nepal has no tested dengue forecast in the peer-reviewed literature. A transparent, honestly benchmarked early-warning tool is a real contribution, and its limits (late peaks, few events, mixed weather value) are the same limits the field as a whole is still working on.

## Claims to treat with caution

| Claim | Status |
|---|---|
| Western Nepal 2026 SARIMAX study was "best" with weather | Search snippet only; metrics not verified |
| Leung 2023: "only 3.9% reported out-of-sample validation" | From a search summary; not found in the full text |
| Benjarattanaporn 2026 network meta-analysis | medRxiv preprint, not peer reviewed |
| Eastin 2014 (Colombia): climate wins at 6 months | Search snippet; full text not read |
| Vietnam "relative value in ~76% of provinces" | Tool summary of full text; not in abstract; not used above |
| Vietnam baseline definition | Described as constant incidence; exact specification not confirmed |
| Bangladesh ML papers claiming R² 0.95, 97% accuracy, LightGBM R² 0.095 | Search snippets; designs not verified; not used as evidence |
| Liu et al. 2025 Bangladesh numbers | Internally inconsistent (MAE > RMSE in one version) |
| DengAI leaderboard (top MAE ~13, rank 100 ~17) | Secondary blog source; leaderboard not verified |
| EWARS sensitivity/PPV figures | Retrospective and tuned on the same data; optimistic |
| Cardenas 2022 EWARS Mexico figures | Preprint and published versions give different numbers |
| Puerto Rico 40% sensitivity figure | Cited second-hand in Badurdeen 2013; original not read |
| Tree-model extrapolation limit; confidence interval for 4 of 6; cause of negative temperature coefficient | The report's own reasoning or calculation, not from a cited study |

## References

1. Acharya BK, Khanal L, Dhimal M. 2025. PLOS ONE 20(4):e0322031. https://doi.org/10.1371/journal.pone.0322031
2. Badurdeen S, et al. 2013. Sharing experiences: towards an evidence-based model of dengue surveillance and outbreak response. BMC Public Health 13:607. https://doi.org/10.1186/1471-2458-13-607 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3697990/)
3. Benitez-Valladares D, Kroeger A, Tejeda GS, Hussain-Alkhateeb L. 2021. PLoS Negl Trop Dis 15(12):e0009261. https://doi.org/10.1371/journal.pntd.0009261
4. Benjarattanaporn P, Adewo D, Sutton A, Lee A, Dodd PJ. 2026. Dengue forecasting models: systematic review and network meta-analysis. medRxiv (preprint). https://doi.org/10.64898/2026.02.18.26346534
5. Bortman M. 1999. Elaboración de corredores o canales endémicos mediante planillas de cálculo. Rev Panam Salud Publica 5(1):1–8. https://iris.paho.org/handle/10665.2/8562
6. Bracher J, Wolffram D, Deuschel J, et al. 2021. Nat Commun 12:5173. https://doi.org/10.1038/s41467-021-25207-0
7. Brady OJ, Smith DL, Scott TW, Hay SI. 2015. Dengue disease outbreak definitions are implicitly variable. Epidemics 11:92–102. https://doi.org/10.1016/j.epidem.2015.03.002
8. Cardenas R, et al. 2022. BMC Infect Dis 22:235. https://doi.org/10.1186/s12879-022-07197-6
9. Carvajal TM, et al. 2018. BMC Infect Dis 18:183. https://doi.org/10.1186/s12879-018-3066-0
10. Chen X, Moraga P. 2025. Trop Med Health 53:52. https://doi.org/10.1186/s41182-025-00723-7
11. Colón-González FJ, et al. 2021. Probabilistic seasonal dengue forecasting in Vietnam. PLoS Med 18(3):e1003542. https://doi.org/10.1371/journal.pmed.1003542
12. Cramer EY, et al. 2022. PNAS 119(15):e2113561119. https://doi.org/10.1073/pnas.2113561119
13. DrivenData. DengAI benchmark. https://drivendata.co/blog/dengue-benchmark
14. Eastin MD, et al. 2014. Am J Trop Med Hyg. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4155567/
15. Faruk O. 2026. Health Sci Rep. https://doi.org/10.1002/hsr2.72207
16. Frontiers in Tropical Diseases. 2023. Commentary on Nepal's 2022 dengue epidemic. https://doi.org/10.3389/fitd.2023.1217939
17. Guo P, et al. 2017. PLoS Negl Trop Dis 11(10):e0005973. https://doi.org/10.1371/journal.pntd.0005973
18. Hussain-Alkhateeb L, et al. 2018. PLoS ONE 13(5):e0196811. https://doi.org/10.1371/journal.pone.0196811
19. Hyndman RJ, Athanasopoulos G. Forecasting: Principles and Practice, 3rd ed. https://otexts.com/fpp3/simple-methods.html
20. Jha et al. 2025. Trop Med Health. https://pmc.ncbi.nlm.nih.gov/articles/PMC12683827/
21. Johansson MA, Reich NG, Hota A, Brownstein JS, Santillana M. 2016. Sci Rep 6:33707. https://doi.org/10.1038/srep33707
22. Johansson MA, Apfeldorf KM, Dobson S, et al. 2019. An open challenge to advance probabilistic forecasting for dengue epidemics. PNAS 116(48):24268–24274. https://doi.org/10.1073/pnas.1909865116
23. Lauer SA, et al. 2018. Prospective forecasts of annual dengue hemorrhagic fever incidence in Thailand. PNAS 115(10):E2175–E2182. https://doi.org/10.1073/pnas.1714457115
24. Leung XY, et al. 2023. A systematic review of dengue outbreak prediction models. PLoS Negl Trop Dis 17(2):e0010631. https://doi.org/10.1371/journal.pntd.0010631
25. Liu B, Hossain MF, Hossain S. 2025. Sci Rep. https://doi.org/10.1038/s41598-025-19752-7
26. Lopez VK, et al. 2024. Challenges of COVID-19 case forecasting in the US. PLoS Comput Biol 20(5):e1011200. https://doi.org/10.1371/journal.pcbi.1011200
27. Lowe R, et al. 2014. Lancet Infect Dis 14(7):619–626. https://doi.org/10.1016/S1473-3099(14)70781-9
28. Lowe R, et al. 2016. Evaluating probabilistic dengue risk forecasts from a prototype early warning system for Brazil. eLife 5:e11285. https://doi.org/10.7554/eLife.11285
29. Lowe R, et al. 2018. PLoS Med 15(7):e1002613. https://doi.org/10.1371/journal.pmed.1002613
30. Makridakis S, Spiliotis E, Assimakopoulos V. 2018. Statistical and machine learning forecasting methods: concerns and ways forward. PLoS ONE 13(3):e0194889. https://doi.org/10.1371/journal.pone.0194889
31. Makridakis S, Spiliotis E, Assimakopoulos V. 2020. The M4 Competition. Int J Forecast 36(1):54–74. https://doi.org/10.1016/j.ijforecast.2019.04.014
32. Makridakis S, Spiliotis E, Assimakopoulos V. 2022. M5 accuracy competition. Int J Forecast 38(4):1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
33. Mordecai EA, et al. 2017. PLoS Negl Trop Dis. https://doi.org/10.1371/journal.pntd.0005568
34. Nat Commun. 2025. Climate variation and serotype competition in Singapore dengue. https://www.nature.com/articles/s41467-025-66411-6
35. Nepal Health Research Council. Review of climate change and surveillance in Nepal. https://nhrc.gov.np/wp-content/uploads/2021/02/Final-review-climate-change-and-survailalcne-in-Nepal.pdf
36. Nguyen V-H, et al. 2022. PLoS Negl Trop Dis 16(6):e0010509. https://doi.org/10.1371/journal.pntd.0010509
37. Reich NG, et al. 2019. PNAS 116(8):3146–3154. https://doi.org/10.1073/pnas.1812594116
38. Roster K, Connaughton C, Rodrigues FA. 2022. Am J Epidemiol 191(10):1803–1812. https://doi.org/10.1093/aje/kwac090
39. Schlesinger M, et al. 2024. Front Public Health 12:1323618. https://doi.org/10.3389/fpubh.2024.1323618
40. Sesay MM, Ngunyi A, Imboga H. 2026. PLOS Glob Public Health. https://doi.org/10.1371/journal.pgph.0005404
41. Shi Y, et al. 2016. Three-month real-time dengue forecast models (Singapore). Environ Health Perspect 124(9):1369–1375. https://doi.org/10.1289/ehp.1509981
42. Singh S, et al. 2026. Optimizing dengue surveillance thresholds in Malaysia. Trop Med Infect Dis 11(8):231. https://doi.org/10.3390/tropicalmed11080231
43. Sylvestre E, et al. 2022. PLoS Negl Trop Dis 16(1):e0010056. https://doi.org/10.1371/journal.pntd.0010056
44. Tuladhar R, Singh A, Varma A, Choudhary DK. 2019. Climatic factors influencing dengue incidence in an epidemic area of Nepal. BMC Res Notes 12:131. https://doi.org/10.1186/s13104-019-4185-4
45. Umaña JD, et al. 2026. Endemic channel parametrization in dengue surveillance (Colombia). JMIR Public Health Surveill 12:e79914. https://doi.org/10.2196/79914
46. Vega T, et al. 2013. Moving Epidemic Method. Influenza Other Respir Viruses 7(4):546–558. https://doi.org/10.1111/j.1750-2659.2012.00422.x
47. Western Nepal SARIMAX study. 2026. BMC Public Health (unverified). https://doi.org/10.1186/s12889-026-29291-z
48. Wu S, Meyer AG, Clemente L, et al. 2025. Ensemble dengue forecasting across 187 locations. PNAS. https://doi.org/10.1073/pnas.2422335122
49. Xu J, et al. 2020. Int J Environ Res Public Health 17(2):453. https://doi.org/10.3390/ijerph17020453
50. Zhao N, et al. 2020. PLoS Negl Trop Dis 14(9):e0008056. https://doi.org/10.1371/journal.pntd.0008056
