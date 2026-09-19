# Limitations

## The benchmark is easy, and the headline number reflects that

On FEBRL true pairs, 7.04 of 8 fields agree on average, and **no** non-matching
pair — none of 106,422 — agrees on 6 or more fields. There are no hard negatives: no siblings at one
address, no name-and-date-of-birth collisions, no families sharing surname and
postcode. Real administrative data has all three. Accuracy near 0.99 is a
property of this dataset, and the degradation curve in D-10 is the honest
summary of the method's behaviour.

## No domain preprocessing

Production NHS linkage requires knowledge a benchmark cannot supply. NHS
England's published pipeline removes records carrying confidentiality codes,
resolves zero-versus-letter-O confusion in postcodes by character position,
replaces invalid postcodes with a sentinel, and filters roughly sixty patterns
of non-name values in name fields — `twin one`, `baby girl`, `unknown`,
`formerly known as`, titles, `nee`. FEBRL contains none of these values, so none
of that handling is exercised or developed here. It is a real gap between this
result and a production one.

## Conditional independence is violated

Field agreement correlations reach 0.896. Correlated fields double-count their
evidence and make the model appear more certain than it is. Reported, not
corrected.

## Blocking is simpler than production practice

Three exact-match passes here against nine rules in NHS England's pipeline,
including Soundex on family name and a year-plus-day key surviving month/day
transposition. D-10 shows this is the binding constraint as data quality falls,
so it is the most consequential simplification in the project.

## Downstream bias magnitude is simulated

The mechanism — that raising a threshold buys precision and pays in
representativeness — is real, and NHS England report finding non-trivial
differences between linkable and unlinkable records on their own data. The
magnitudes in D-09 come from a simulated outcome and are illustrative only.

## Comparison levels are binary

Each field agrees or disagrees. Production implementations use graded levels
(exact, near, partial, disagree), which carry more information, particularly
for names. Missing values are treated as disagreement, which is conservative
but discards the distinction between "different" and "unknown".

## Not evaluated against an incumbent

NHS England evaluate against MPS, the deterministic service already in use, and
assess differences manually. There is no incumbent here, so the comparison is
against deterministic rules of my own construction.

## Nothing here is NHS data

FEBRL is synthetic. No NHS record was used, and no result transfers as a
number.
