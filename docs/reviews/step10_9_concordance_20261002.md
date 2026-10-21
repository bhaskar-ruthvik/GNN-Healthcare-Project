# PCP–specialist race concordance (Amendment A13, 2026-10-02)

Race from `pcp_race` / `ren_race` (computed/imputed in the source). Provider gender not analysed (inferred values inconsistent). Concordant = specialist's race equals the PCP's race. Availability = exact expectation of a uniform random pick among the eligible specialists. PCP-cluster bootstrap. Outputs `runs/analysis/step10_9_concordance/{split}_concordance.csv`, `concordance_{split}.png`.

| All PCPs with known race (top pick) | 2018 (77 PCPs) | 2019 (49 PCPs) |
|---|---|---|
| Availability (random eligible specialist) | 28.6% | 27.0% |
| Actual referrals | 32.9% (excess +4.2) | 31.1% (excess +4.1) |
| PCP’s habit | 35.5% (+6.9) | 35.1% (+8.1) |
| Model’s own ranking | 35.6% (+7.0) | 34.2% (+7.1) |
| Model + 30% quality weight | 32.5% (+3.8) | 31.0% (+3.9) |
| + load penalty 0.2 | 29.9% (+1.3; change vs practice −3.0 [−5.9, −0.1]) | 30.2% (+3.1; −1.0 [−5.2, +3.4]) |

- Practice shows mild same-race matching: ~4 points more concordant than availability alone explains.
- The model’s own ranking (imitating PCP habits) matches slightly more (+3 points vs practice; not significant overall; significant for White PCPs on 2018: +6.5 [1.5, 8.8]).
- Adding the 30% quality weight brings matching back to the practice level; the load penalty reduces it further (significant on 2018 only).
- By PCP race: Asian PCPs have the largest practice excess on 2019 (+16 points); the quality-weighted lists reduce it (−7, interval includes 0). Black and Hispanic PCP groups are too small on 2019 (< 10 PCPs).
Caveats: few PCPs (49–77 with known race), so intervals are wide; race is imputed; concordance can reflect referral networks, language and geography rather than bias; descriptive only.
