# Disease-category inspection — 2026-09-21

Read-only scan of all 608,181 rows in the source workbook's Final_Base_Data_09_30_2025 sheet. Aggregates split by DOC-DOC-LINK; legacy negatives are not proposed as training examples. No patient-level records exported. Counts below are source rows, not reconstructed encounters.

Observed positive rows: 273,711. D_Disease_Code_Category has 33 distinct nonempty labels and 132 blank rows (99.95% nonblank coverage). This is not a six-category field as stored.

D_Disease_Code_Severity has Simple (113,018), Moderate (98,300), Complex (62,261), and 132 blanks. These severity labels are not evidence of the meaning of the handwritten three duration bands.

Each raw diagnosis string maps to a single raw category in this scan (zero conflicting diagnosis-to-category mappings within positives or negatives). This checks consistency of existing assignments, not clinical correctness, coding-version validity, derivation provenance, or pre-referral availability. Encounters may have multiple diagnosis codes and therefore multiple categories; preserve that multiplicity in subsequent input preparation.

Some category names overlap or differ by wording, including 'and' versus '&', and there are combined categories such as 'Neoplasms / Blood & immune disorders'. Do not automatically collapse labels or invent six groups without a verified crosswalk. Temporary ICD9/ICD10 category columns exist; their metadata designates them temporary. The final field should be the initial source candidate.

Recommended first structural comparison: use the same clinical representation in both three-node arms. A switch from raw codes to category inputs should be explicitly versioned and applied to both arms, with training-only vocabulary and unknown handling. Do not change event grouping solely because this field exists. Fit statistical preprocessing only on training data; this whole-source scan was a schema/coverage inspection, not model fitting.

Aggregate details: disease_category_audit_20260921.json. No prepared inputs, source workbook, notebook, or model weights changed.
