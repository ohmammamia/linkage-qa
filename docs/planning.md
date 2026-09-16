# Planning

Written before the data was opened.

## The problem

Two administrative datasets describe overlapping populations. Neither carries a
shared reliable identifier. Records must be resolved to people accurately
enough that downstream analysis is trustworthy, at a scale where comparing
every possible pair is infeasible, while quantifying how much of the result is
uncertain.

## The user

An analyst building a linked dataset that other people will analyse, working
with a data manager who owns the clerical review capacity.

They decide three things: which blocking strategy to run, where to set the
acceptance threshold, and how much manual review to buy. The analysis exists to
support those three decisions and is structured around them.

## Cost of the two errors, stated before any result

These are not symmetric, and no single metric expresses the difference.

**A false match** fuses two people into one record. The resulting person has
someone else's history attached. In healthcare this is a safety problem before
it is a statistical one, and it is invisible downstream: the record looks
complete and internally consistent.

**A missed match** splits one person into two. Their history is incomplete,
cohorts shrink, and the loss is silent unless someone measures it.

Which to prefer depends on use. A safety or care-delivery application should
fear false matches more. A prevalence study should worry about missed matches,
because the people hardest to link are systematically different from the people
easiest to link. Since the use is not fixed here, thresholds are reported as a
curve with counts of each error rather than a single recommended value.

## What "good" means, declared in advance

1. Blocking retains at least 99% of true pairs while removing at least 99% of
   candidate comparisons. Both, together.
2. The probabilistic model beats the best deterministic rule on end-to-end
   recall at equal precision.
3. Model fitting uses no labels. Labels evaluate; they never train. Otherwise
   the evaluation measures nothing that transfers to production, where no
   labels exist.
4. Every quality check in the QA framework is computable without ground truth.

## Known constraint

The benchmark is FEBRL, which is synthetic with controlled corruption. Error
rates here are not NHS error rates. What transfers is the method, the
diagnostics and the shape of the trade-offs, not the numbers.
