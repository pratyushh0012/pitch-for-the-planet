# Do climate/weather variables improve dengue forecasts? Evidence for and against

Note on method: about 18 search/fetch calls. Several publisher pages (Nature, PNAS, Wiley, ResearchGate) returned 403 or redirects, so some findings come from abstracts or search snippets. These are flagged. Two kinds of study are kept apart throughout:
- **(A) Association/explanatory studies:** climate is correlated with dengue in-sample.
- **(B) Out-of-sample forecast/ablation studies:** a model with climate is compared against a model using only past cases and/or seasonality.

## 1. Ablation-style forecast studies: does adding climate beat past cases plus seasonality?

### Takeaway
The results are mixed and depend on context. The cleanest ablation, Johansson et al. 2016 for Mexico, found that climate data did **not** significantly improve seasonal autoregressive models; autocorrelation did most of the work. Studies that report gains from climate (Lowe et al. in Brazil, Barbados and Vietnam) usually compare against a weaker "null/seasonal-average" baseline, not a strong autoregressive baseline. Gains are larger for outbreak or season-level targets and at some lead times, and they vary by location.

### Cited Findings
- **Johansson, Reich, Hota, Brownstein & Santillana (2016), *Sci Rep* 6:33707 (Mexico), doi:10.1038/srep33707.** Compared climate-driven and seasonal dengue forecasts for Mexico. Climate data "did not significantly improve the predictive power of seasonal autoregressive models". Short-term autocorrelation was key for short-horizon forecasts and seasonal autocorrelation for long-horizon forecasts. Seasonal autoregressive models captured "a substantial amount of dengue variability", but better models are still needed — [Nature/Sci Rep abstract via search](https://www.nature.com/articles/srep33707); [citation record](https://www.mindat.org/reference.php?id=4655677). (The full text could not be fetched, so exact per-state counts are not verified.)
- **Johansson et al. (2019), *PNAS*, "An open challenge to advance probabilistic forecasting for dengue epidemics" (the Dengue Forecasting Project: Iquitos, Peru and San Juan, Puerto Rico).** Ensemble forecasts beat most individual forecasts and the null forecast for all targets across 8 seasons. Teams using ensemble approaches had higher log scores — [PNAS](https://www.pnas.org/doi/10.1073/pnas.1909865116). I could not retrieve the paper's exact statement on climate-using versus non-climate models because the full text was blocked (403). As commonly summarised, the paper reported that models using climate/environmental data did not clearly outperform those that did not. **This is unverified here and should be checked before quoting.**
- **Colón-González et al. (2021), *PLoS Med* 18(3):e1003542 (Vietnam, superensemble of Bayesian spatiotemporal negative binomial models), doi:10.1371/journal.pmed.1003542:**
  - The climate-informed superensemble beat the baseline at 1–3-month leads: CRPS 66.8 (95% CI 60.6–148.0) versus 79.4 for the baseline.
  - Skill was about 42% better than baseline at a 1-month lead (CRPSS ≈ 0.42) but **12.5% worse than baseline at a 6-month lead**.
  - Outbreak detection was 69% versus 54.5% for the baseline.
  - Key predictors were temperature, precipitation, specific humidity and diurnal temperature range centred at a 1-month lag, plus lagged sea-surface-temperature (ENSO-type) anomalies — [PLoS Med](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1003542).
  - Caveat: the baseline's exact specification (seasonal random effects only?) was not confirmed in my fetch.
- **Lowe et al. (2016), *eLife*, evaluation of the Brazil 2014 World Cup early warning system** (driven by real-time seasonal climate forecasts). For high dengue risk, the forecast model had a 57% hit rate versus 33% for a null model based on seasonal averages of past incidence — [eLife](https://elifesciences.org/articles/11285); [PMC4775211](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4775211/). The original forecast was Lowe et al. 2014, *Lancet Infect Dis* — [PubMed 24841860](https://pubmed.ncbi.nlm.nih.gov/24841860/). Note that the null model was a seasonal average, not an autoregressive model built on recent cases.
- **Lowe et al. (2018), *PLoS Med* 15(7):e1002613 (Barbados; distributed lag non-linear model, DLNM), doi:10.1371/journal.pmed.1002613:**
  - Drought (negative 6-month Standardised Precipitation Index, SPI-6) raised dengue risk with a peak effect about 5 months later. Excess rainfall raised risk at 1–2-month lags.
  - "A drought period followed by intense rainfall 4 to 5 months later could provide optimum conditions for an imminent dengue outbreak."
  - The final model correctly classified outbreak versus non-outbreak months 86% of the time (81–91%), versus 64% (58–71%) for the baseline model. Hit rate 90%, false alarm rate 31%, AUC 0.90.
  - Performance broke down in 2014–2016, when chikungunya and Zika co-circulated — [PLoS Med](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002613).
- **Eastin et al. (2014), *Am J Trop Med Hyg* (Cali, Colombia), "Intra- and interseasonal autoregressive prediction of dengue outbreaks using local weather and regional climate":**
  - Climate-based models **underperformed autoregressive models at 1- and 3-month leads and outperformed them at a 6-month lead** — [PMC4155567](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4155567/) (from a search snippet; fetching the full text failed).
  - A related finding in the search results: autoregressive predictors drop in percent inclusion, and global climate features rise, as lead time grows. These patterns were city-specific (same snippet set; attribution not fully verified).
- **Shi et al. (2016), *Environ Health Perspect* 124(9):1369–1375 (Singapore):**
  - LASSO models with more than 200 candidate variables: recent cases, weather, vector surveillance, population and seasonality.
  - Weekly forecasts for 1–12 weeks ahead. LASSO beat SARIMA and stepwise linear regression on mean absolute percentage error — [EHP](https://pubs.acs.org/evhpaz/article/124/9/1369/5245100/Three-Month-Real-Time-Dengue-Forecast-Models-An); [EHP news summary](https://pubs.acs.org/evhpaz/article/124/9/A167/5245111/Singapore-Success-New-Model-Helps-Forecast-Dengue).
  - The paper does not cleanly isolate the contribution of climate; it compares a multi-stream model against time-series baselines.
- **Stolerman, Maia & Kutz (2019), *PLoS ONE* (Brazil):** support vector machines applied to climate variables tied to the mosquito life cycle, used to classify epidemic versus non-epidemic years across Brazilian cities — [ResearchGate record](https://www.researchgate.net/publication/335076650_Forecasting_dengue_fever_in_Brazil_An_assessment_of_climate_conditions). I could not retrieve accuracy figures or the city-by-city breakdown (403). Follow-up work added population-susceptibility cycles to weather to forecast epidemic years in Brazil — [PMC8205538](https://pmc.ncbi.nlm.nih.gov/articles/PMC8205538/). This suggests that climate alone was not enough.
- **Large incidence-only benchmarks.** A 2025 *PNAS* global ensemble evaluation covered 11 model types and more than 180 locations in Brazil, Colombia, Malaysia, Mexico and Thailand. Apart from ARGO/ARGONet, it trained all models **only on historical case counts**. It still produced useful short-term forecasts. The authors note that "localized climate shocks that fall outside the recent training window may therefore elude prediction", but they did not test climate empirically — [PMC12377650](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/).
- **Beal et al. (2025), *GeoHealth*, "Forecasting Dengue: Evaluating the Role of Hydroclimate Information in Subseasonal to Seasonal Prediction":** directly targets the question of climate's added value by lead time — [Wiley](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GH001325). The full text was blocked (403), so the results are not verified.
- **Other recent benchmarks:** the 2024 Brazil Dengue Forecasting Sprint (Mosqlimate, *PNAS*) — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2508989123); [Mosqlimate arXiv](https://arxiv.org/pdf/2410.18945). There is also a 2026 medRxiv systematic review and meta-analysis of dengue forecasting models, which reports k-NN, VAR, Kalman filtering, GLM and autoregressive neural networks as the most accurate classes — [medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.02.18.26346534v1.full.pdf). Neither result on the specific climate ablation was verified.

### Inferences
- When the baseline is strong (autoregressive on recent cases plus seasonality), adding climate often gives little or no gain at 1–4-week or 1-month horizons (Mexico; Colombia at 1–3 months). Gains are more often reported when:
  - the baseline is a seasonal climatology or null model (Brazil eLife, Barbados);
  - the target is outbreak/epidemic-year classification;
  - the lead time is longer, about 3–6 months (Colombia at 6 months). Vietnam is the exception: there the gains were at 1–3 months and turned negative by 6 months.
- So the evidence on long leads conflicts: Colombia found climate helps at 6 months, while Vietnam found the climate superensemble fell below baseline at 6 months. The value at long leads is therefore not guaranteed. It probably depends on the quality of the seasonal climate forecasts and on ENSO teleconnection strength in each place.
- A rigorous evaluation should ablate against an autoregressive plus seasonal baseline under rolling-origin out-of-sample validation, and report results per location and per lead time.

### Gaps
- **HU Berlin thesis/paper finding that climate is city-dependent:** I could not locate it. A search for a Berlin thesis returned nothing. Do not cite it without the source.
- **Chen et al.:** too ambiguous (many "Chen et al." dengue papers, e.g. Chen Y. et al. 2018 Singapore/Malaysia models), and none was verified here.
- **Exact climate-versus-non-climate statement in the 2019 PNAS Dengue Forecasting Project paper:** not verified (403).
- **Johansson 2016:** exact metrics (e.g. the number of states where climate helped) not verified.
- **Lowe et al. Ecuador (Machala; 2017 *Lancet Planet Health* forecast of the 2016 season):** not fetched in this session.

## 2. Nepal / South Asia specific evidence

### Takeaway
Nepal evidence is mostly associational (type A), not forecast ablation. In Chitwan, minimum temperature lagged 2 months and current relative humidity were significant, while rainfall dropped out of the model. Separate work documents rising thermal suitability, and Aedes establishment, at mid-hill altitudes of 500–1,500 m and above. This strongly implies that national-average weather mixes different climate regimes.

### Cited Findings
- **Tuladhar R, Singh A, Varma A, Choudhary DK (2019), "Climatic factors influencing dengue incidence in an epidemic area of Nepal", *BMC Res Notes* 12:131, doi:10.1186/s13104-019-4185-4.**
  - Design: monthly cases in Chitwan district, 2010–2017, analysed with negative binomial regression.
  - The final model (Model 4) kept four variables:

    | Variable | Lag | IRR (per °C or per % point) |
    |---|---|---|
    | Minimum temperature (strongest predictor) | 2 months | 1.437 |
    | Maximum temperature | 0 | 1.193 |
    | Relative humidity | 0 | 1.109 |
    | Maximum temperature (inverse effect) | 3 months | 0.759 |

  - Rainfall correlated with cases at 1–3-month lags but **was not retained as a significant predictor** in the final model.
  - The authors recommend vector control from June–July.
  - Association study only; there was no out-of-sample forecast test — [PMC6417253](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417253/).
- **Acharya BK, Khanal L, Dhimal M (2025), *PLOS ONE* 20(4):e0322031, doi:10.1371/journal.pone.0322031:**
  - Method: satellite land surface temperature for 2001–2020 fed into mechanistic R0 models. Validation AUC was above 0.75 for both vector species.
  - About half of Nepal is thermally suitable for at least 1 month per year; lowlands reach up to 9 months.
  - Suitability expanded most in the **500–1,500 m** zone.
  - From 2010 to 2020, the share of the population exposed to 6 or more suitable months rose 18% for *Ae. aegypti* and 20% for *Ae. albopictus*.
  - By 2020, about 93% of residents lived in areas suitable for at least 1 month. Kathmandu and Pokhara now have up to 6 suitable months — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322031).
- **Altitudinal vector genomics, Nepal (Molecular Ecology, 2022/23):** *Ae. aegypti* was sampled between 200 and 1,300 m. The highland population at 1,300 m was genomically distinct from lowland populations (800 m or lower), with allele differences in climate-adaptation genes. This points to non-gradual expansion and possible cold adaptation — [Mol Ecol](https://onlinelibrary.wiley.com/doi/10.1111/mec.16752); [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.04.20.488929v1).
- **Himalayan warming:** temperatures in the Himalayas are rising about 0.2°C per decade, with more days in Nepal falling within the 20–30°C range ideal for *Ae. aegypti* — [Science News](https://www.sciencenews.org/article/nepal-reeling-from-unprecedented-dengue-virus-outbreak) (journalistic source). A Hindu Kush Himalaya systematic synthesis reports vectors establishing above 1,500 m and up to about 2,438 m — [ScienceDirect HKH review](https://www.sciencedirect.com/science/article/pii/S1674927821000757); also [Foreign Policy 2023](https://foreignpolicy.com/2023/10/06/nepal-disease-dengue-climate-change-global-warming-himalayas/).
- **Bangladesh:** a time-series study links dengue incidence to meteorological risk factors — [PMC12360610](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12360610/). A deep-learning climate-dengue model also exists — [PLOS Glob Public Health](https://journals.plos.org/globalpublichealth/article?id=10.1371%2Fjournal.pgph.0006405). Neither was fetched, so no numbers are verified.

### Inferences
- Nepal's dengue-climate signal is strongly elevation-dependent: Terai lowlands versus mid-hills versus mountains. A national-average temperature or rainfall series would blend regimes with very different thermal suitability, diluting or distorting the lagged signal. District or eco-zone climate covariates are more defensible.
- The Chitwan finding (minimum temperature at a 2-month lag matters; rainfall does not) fits the mechanism that cool-season minimum temperatures limit vector and virus development near the thermal floor.

### Gaps
- No Nepal study was found that tests the **out-of-sample added value** of climate over a cases-only model.
- The original Dhimal et al. papers (e.g. 2015 *PLoS ONE* on Aedes altitudinal distribution; 2015 *PLoS NTD*/BMC on dengue-climate links in Nepal) were not fetched. The Dhimal-coauthored 2025 *PLOS ONE* paper stands in for them.
- No verified details were found on Dhaka forecasting skill or on Sri Lanka climate-dengue forecasting (e.g. Colombo).

## 3. Mechanisms, lags and nonlinearity

### Takeaway
Temperature affects transmission nonlinearly. The thermal optimum is about 26–29°C, and transmission is possible between roughly 18 and 34°C. Warming speeds up the extrinsic incubation period (EIP). Together with vector development, these delays create the typical 1–3-month lag between weather and cases. Rainfall can act in opposite directions at different lags (drought about 5 months before, excess rain 1–2 months before). Linear, single-lag models can therefore miss the signal.

### Cited Findings
- **Mordecai et al. (2017), *PLoS NTD*, "Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models":** dengue transmission peaks at 26–29°C and can occur between about 18 and 34°C. Species-specific optima: *Ae. aegypti* 29.1°C (range 17.8–34.5°C); *Ae. albopictus* 26.4°C (range 16.2–31.4°C) — [PLoS NTD](https://journals.plos.org/plosntds/article?id=10.1371%2Fjournal.pntd.0005568); see also [Mordecai et al. 2019 *Ecol Lett*](https://onlinelibrary.wiley.com/doi/10.1111/ele.13335).
- **Chan M & Johansson MA (2012), "The Incubation Periods of Dengue Viruses", *PLoS ONE* 7(11):e50972.** The EIP in the mosquito follows a log-normal distribution. Both its mean and its variance shrink as temperature rises. Many later temperature-dependent EIP models use their estimates — [ResearchGate](https://www.researchgate.net/publication/233889049_The_Incubation_Periods_of_Dengue_Viruses). (Specific day values, about 15 days at 25°C and about 6–7 days at 30°C, come from memory of the paper and were not verified this session.)
- **Nonlinear meta-analysis:** temperature effects on dengue incidence are nonlinear and modified by climatic and socioeconomic factors — [PLOS Climate meta-analysis](https://journals.plos.org/climate/article?id=10.1371%2Fjournal.pclm.0000152). Temperature-dependent generation intervals also change the effective reproduction number — [PMC5536440](https://pmc.ncbi.nlm.nih.gov/articles/PMC5536440/).
- **Lags in forecast and association models:**
  - Vietnam: climate predictors centred at a 1-month lag ([Colón-González 2021](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1003542)).
  - Chitwan: minimum temperature at 2 months, with an inverse maximum-temperature effect at 3 months ([Tuladhar 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417253/)).
  - Barbados: drought peak at about 5 months, excess rain at 1–2 months, modelled with a DLNM ([Lowe 2018](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002613)).
- **Singapore (2025, *Nat Commun*):** climate variation together with serotype competition drives dengue outbreak dynamics. Climate is not the only driver; immunity and serotype shifts matter too — [Nature Comms](https://www.nature.com/articles/s41467-025-66411-6).

### Inferences
- Because the effects are nonlinear (inverted-U temperature, rainfall acting in both directions), forecasting models should use splines, DLNMs or tree-based methods rather than linear terms. A hot location already above the optimum may see warming *reduce* risk.
- Lags of 1–3 months mean climate observed now mostly informs cases 1–3 months ahead. Within the next 1–4 weeks, its effect is largely already reflected in current case counts.

### Gaps
- No verified numbers were obtained for the Naish et al. 2014 (*BMC Infect Dis*) climate-dengue systematic review or the Ebi & Nealon 2016 (*Environ Res*) review.

## 4. Why case history dominates at short horizons, and when climate helps more

### Takeaway
At short horizons (weeks to about 1 month), recent case counts already contain the effect of climate from 1–3 months earlier, plus immunity and serotype dynamics. So autoregressive and incidence-only models are hard to beat. Climate adds most at longer leads, for predicting season size or outbreak years before the season starts (especially with ENSO or seasonal climate forecasts), and in places where climate drives large year-to-year variation.

### Cited Findings
- In Mexico, short-term autocorrelation drove short-term skill and seasonal autocorrelation drove long-term skill; climate added no significant improvement — [Johansson 2016](https://www.nature.com/articles/srep33707).
- In Colombia, climate models lost to autoregressive models at 1–3-month leads but won at 6 months — [PMC4155567](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4155567/).
- In Vietnam, the climate superensemble was about 42% better than baseline at 1 month but 12.5% worse at 6 months. This contradicts the simple rule that climate always helps more at longer leads — [Colón-González 2021](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1003542).
- Season-ahead outbreak warnings driven by climate forecasts beat seasonal-average baselines in Brazil (57% vs 33% hit rate) and Barbados (86% vs 64% accuracy) — [eLife 2016](https://elifesciences.org/articles/11285); [PLoS Med 2018](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002613).
- Large incidence-only ensembles across more than 180 locations gave competitive short-term forecasts without climate — [PNAS 2025 / PMC12377650](https://pmc.ncbi.nlm.nih.gov/articles/PMC12377650/).
- Population susceptibility and serotype dynamics also determine epidemic years, so climate alone is insufficient — [PMC8205538](https://pmc.ncbi.nlm.nih.gov/articles/PMC8205538/); [Nat Commun 2025](https://www.nature.com/articles/s41467-025-66411-6).

### Inferences
- **For a pitch or product:** present climate as most valuable for (1) seasonal outlooks and preparedness 2–6 months ahead, (2) pre-season estimates of epidemic size, and (3) settings where transmission is at the thermal edge, such as Nepal's mid-hills where suitability is expanding. Do not present it as a guaranteed boost to week-ahead nowcasts.
- **Honest framing:** "climate improves forecasts in some places and at some horizons, and evaluation must be done per location." This is better supported than a blanket claim.

### Gaps
- No quantitative meta-analysis of the "skill gain from climate by lead time" across studies was found.
- The Beal et al. 2025 *GeoHealth* paper may provide this, but it was not accessible.
