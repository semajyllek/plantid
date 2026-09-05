"""The community tool: turn a chosen species list into a small offline model.

Three commands, and the ordering is the point:

    plantid plan    what will this set give me, and why    (no training)
    plantid build   fetch -> embed -> fit -> export
    plantid card    the honest report on a built model

`plan` exists because the measurements in `EMBEDDED_FINDINGS.md` show that the
*composition* of a species list determines which failure mode it has, and a user
cannot predict that themselves. A congener-dense set does not answer wrongly, it
answers vacuously -- 31% species-level rather than 83% -- while its coverage and
precision look *better* than a well-separated set. Reporting coverage without the
species-level share would tell that user their worst case is their best case.
"""
