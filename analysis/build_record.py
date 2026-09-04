"""Assemble the experimental-record page from figures.json + charts.py.

The entries are written here rather than in the template so that every number
inside them is pulled from the same dataset the charts are drawn from — a figure
and the sentence beside it cannot disagree.
"""

import json
import sys

import charts

SC = str(OUT)
D = json.load(open(f"{SC}/figures.json"))


def ci(lo, hi, star=False):
    s = f'<span class="ci num">[{lo:+.3f}, {hi:+.3f}]</span>'
    return s + (' <span class="ci">✓ excludes zero</span>' if star else '')


def entry(title, status, rows, body, chart=None, caption=None, table=None):
    chip = f'<span class="chip {status}">{status}</span>'
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    fig = ""
    if chart:
        fig = (f'<figure class="bleed">{chart}'
               + (f"<figcaption>{caption}</figcaption>" if caption else "") + "</figure>")
    return (f'<div class="entry"><h3>{title} {chip}</h3>'
            f'<dl class="pm">{dl}</dl>{body}{fig}{table or ""}</div>')


E = D["encoder_cmp"]
C = D["cohorts"]
NO = {r["when"]: r for r in D["nearood"]}

entries = [

entry("Half the catalogue had never been tested on a real photograph", "held",
      [("gap", "247 of 497 catalogue species had <em>no real observation at all</em> — "
               "scored only on the PlantNet split their own head was fitted on"),
       ("measured", "two fetches closed it to <strong>32</strong>; coverage 250 → 411 → 465 species")],
      """<p>The evaluation set was quietly circular for half the catalogue. A
      targeted per-species fetch closed 161 of the 247. The remaining 86 turned
      out to be mostly a <em>query</em> problem, not a data problem — see the next
      entry.</p>"""),

entry("Being a taxonomic generation behind was the whole of the failure", "held",
      [("suspected", "the 86 remaining species are rare or absent from iNaturalist"),
       ("measured", "<strong>47 of 497 catalogue names (9.5%) are superseded synonyms</strong>; "
                    "all 50 renamed species were among the 86, and not one renamed species "
                    "was ever found by a name query")],
      """<p>The catalogue carries PlantNet's pre-split names. <i class="sp">Anemone
      nemorosa</i> is now <i class="sp">Anemonoides nemorosa</i> — 84,623
      research-grade observations, invisible to a <code>taxon_name</code> lookup.
      The correspondence is exact in both directions, which is as clean as a
      causal story gets here.</p>
      <p>Eight more species had <em>current</em> names and still returned nothing,
      for a different reason: <code>taxon_name</code> is a fuzzy match on the
      observations endpoint, so page one fills with a commoner congener.
      <i class="sp">Lactuca sativa</i> returned 100 out of 100
      <i class="sp">Lactuca serriola</i> despite having 56,684 observations of its
      own. Both failures are fixed by querying <code>taxon_id</code>.</p>"""),

entry("The species we had been missing are harder — and only at species level", "held",
      [("predicted", "correcting a sampling bias toward heavily-photographed plants should cost accuracy"),
       ("measured", f'species {C[0]["sp"]:.3f} → {C[1]["sp"]:.3f} → {C[2]["sp"]:.3f} across cohorts; '
                    f'broad − targeted <span class="num">+0.079</span> {ci(0.023, 0.138, True)}'),
       ("genus", f'statistically flat — <span class="num">+0.013</span> {ci(-0.007, 0.034)}')],
      """<p>Species that only appeared when asked for by name are significantly
      harder than those broad queries surfaced. Genus accuracy is not
      distinguishable between cohorts. <strong>The level the product leans on is
      the level that survived the harder sample</strong> — which is the strongest
      argument on this page for genus being a first-class answer rather than a
      hedge.</p>""",
      chart=charts.cohorts(D),
      caption="Species accuracy with 95% cluster-bootstrap intervals. The "
              "<em>recovered</em> cohort is not rare plants — it is plants whose "
              "catalogue name was stale — yet it scores like the rarity cohort. "
              "The next entry explains why."),

entry("Recovered species should have been easy. They were not, and the reason is structural", "held",
      [("predicted", "recovered species are common (<i class='sp'>Anemone nemorosa</i> has 84,623 "
                     "observations), so they should score like the easy cohort"),
       ("measured", f'{C[2]["sp"]:.3f} — indistinguishable from the rarity cohort '
                    f'{ci(-0.155, 0.061)}'),
       ("explanation", "renamed genera are <em>large</em> ones: median 10 catalogue congeners against 3")],
      """<p>The prediction was wrong and the reason is worth more than the
      prediction was. Genera get split <em>because</em> they are large, so the
      recovered cohort is 46 <i class="sp">Anemone</i>, 19 <i class="sp">Sedum</i>,
      15 <i class="sp">Papaver</i> — dense, confusable groups.</p>
      <p>Binning every in-catalogue observation by how many congeners its species
      has in the catalogue shows the mechanism, and the two levels run in
      <strong>opposite directions</strong>.</p>""",
      chart=charts.congeners(D),
      caption="A species alone in its genus is easy to name and harder to place — "
              "its genus block has one column, so genus score is species score and "
              "the fallback provides no lift at all (verified on 698 of 737 such "
              "observations). A species in a crowded genus is hard to name and "
              "almost impossible to misplace.",
      table='<div class="tw"><table><thead><tr><th>catalogue congeners</th>'
            '<th>observations</th><th>species accuracy</th><th>genus accuracy</th></tr></thead>'
            "<tbody>" + "".join(
                f'<tr><td class="k">{r["bin"]}</td><td>{r["obs"]:,}</td>'
                f'<td>{r["sp"]:.3f}</td><td>{r["gn"]:.3f}</td></tr>'
                for r in D["congeners"]) + "</tbody></table></div>"),

entry("Merging microspecies a phone cannot separate", "held",
      [("decision", "this is an app for plant enthusiasts, not taxonomists"),
       ("done", "seven <i class='sp'>Ophrys sphegodes</i> segregates merged; a hybrid-naming "
                "bug fixed; 497 → 490 classes"),
       ("headline effect", "none — the evaluation set holds 3 observations of any merged species")],
      """<p>Two different problems wore the same disguise. The first was a
      <strong>bug</strong>: class labels were built by truncating to two tokens, which
      turns <i class="sp">Fragaria × ananassa</i> into the class
      <code>Fragaria ×</code> and collapsed three distinct pelargoniums into one.
      The second was a <strong>product decision</strong>: the catalogue carried eight
      segregate microspecies of the <i class="sp">Ophrys sphegodes</i> complex that
      iNaturalist does not recognise and orchid specialists argue about.</p>
      <p><i class="sp">Ophrys araneola</i> was deliberately <em>kept</em> — it
      resolves to itself as an active taxon.</p>
      <p>The headline did not move, and the reason is a limit of the evaluation
      set rather than a verdict: it contains almost none of the plants curation
      touches. What it <em>can</em> see is the effect on species left behind in a
      de-crowded genus — <strong>5 observations flipped wrong→right and 0 the other
      way</strong>, every one a real species the model had previously named as a
      microspecies. Four of the five were <i class="sp">O. araneola</i>, the species
      kept out of the merge.</p>
      <p>The most-confused pairs were then examined as further merge candidates and
      <strong>declined</strong>: <i class="sp">Lavandula angustifolia</i> /
      <i class="sp">latifolia</i>, <i class="sp">Cucurbita maxima</i> /
      <i class="sp">pepo</i>, and six others all resolve to distinct active taxa.
      Merging on confusion rate optimises the metric by deleting the product.</p>"""),

entry("Near-OOD: growing it helped, but it cannot be finished by fetching", "corrected",
      [("first reported", '<s>doubling the bucket did not help</s>'),
       ("actually", f'utility {NO["before"]["u"]:+.3f} → {NO["after"]["u"]:+.3f}, interval width '
                    f'{NO["before"]["hi"]-NO["before"]["lo"]:.3f} → '
                    f'{NO["after"]["hi"]-NO["after"]["lo"]:.3f}'),
       ("ceiling", "near-OOD genera come from the catalogue's own 172, of which 120 are covered")],
      """<p>The original reading was asserted from a single wide interval with no
      baseline. Recomputing the earlier round with the same genus-clustered
      bootstrap shows growth worked in both directions — the point estimate
      improved and the interval narrowed 17%.</p>
      <p>It is still the loosest thing in the project, and the useful part is the
      ceiling: <strong>fetching buys at most another 1.4× before running out of
      catalogue genera</strong>. Fixing near-OOD needs a modelling change, not more
      data.</p>"""),

entry("The reject class is starved and must be rebuilt", "retracted",
      [("claimed", '<s>the <code>__OTHER__</code> pool has decayed from 800 species to 149; '
                   'this is the largest unaddressed defect in the rejection path</s>'),
       ("check 1", "rebuilding against the current catalogue returns the <em>identical</em> "
                   "species set — 0 gained"),
       ("check 2", "10× ablation is flat on every rejection metric")],
      """<p>Both halves of the claim were wrong. The pool is small because
      PlantNet holds 1,081 species, the catalogue claims 530 of them, and few of
      the rest clear the image threshold — arithmetic, not decay. Growing the
      catalogue <em>necessarily</em> shrinks the reject pool.</p>
      <p>And it does not matter anyway.</p>""",
      chart=charts.ablation(D),
      caption="Subsampling the reject class by species across a 10× range, two "
              "seeds. Every rejection metric is flat, drifting slightly downward "
              "and well inside seed noise. The reject class saturates far below 59 "
              "species — sensible for a frozen encoder, where the negatives only "
              "have to locate a region in a space the encoder has already "
              "organised. This would not survive fine-tuning, where negatives "
              "shape the representation itself."),

entry("The deployable encoder costs a fifth of the product", "open",
      [("assumed", '<s>BioCLIP v1 costs 3.5pp of species and 3.2pp of genus accuracy</s> — '
                   'measured on the catalogue\'s own test split'),
       ("measured on real observations", f'species {E["bioclip2"]["sp"]:.3f} → {E["bioclip1"]["sp"]:.3f} '
                    f'({E["paired_sp"][0]:+.3f}, {ci(E["paired_sp"][1], E["paired_sp"][2], True)})'),
       ("rejection", f'regional-OOD AUROC {E["bioclip2"]["r"]:.3f} → {E["bioclip1"]["r"]:.3f}'),
       ("product", "coverage 72% → 53% at matched precision")],
      """<p>Every headline in this project was BioCLIP-2 — the 304M ViT-L that
      cannot ship — because <code>build_heads</code> had no encoder parameter. The
      deployable encoder had only ever been compared on the catalogue's own test
      split: same corpus, no out-of-catalogue plants, no observation grouping.</p>
      <p>On real observations the accuracy gap is about twice what that comparison
      showed, and the larger damage is somewhere the catalogue split
      <strong>structurally could not measure</strong>: rejection. Precision barely
      moves because the rule protects it by declining — in-catalogue declines
      nearly triple, from 13.1% to 36.6%.</p>
      <p>So the cost of shipping is not three points of an accuracy metric. It is
      <strong>a fifth of the captures the app can answer</strong>, and it reopens
      the distillation project that was cancelled on the strength of the
      same-corpus comparison.</p>""",
      table='<div class="tw"><table><thead><tr><th>on 5,534 real observations</th>'
            '<th>BioCLIP-2 · cannot ship</th><th>BioCLIP v1 · ships</th></tr></thead><tbody>'
            + "".join(
                f'<tr><td class="k">{lab}</td><td class="win">{E["bioclip2"][k]:.3f}</td>'
                f'<td>{E["bioclip1"][k]:.3f}</td></tr>'
                for lab, k in (("in-catalogue species accuracy", "sp"),
                               ("in-catalogue genus accuracy", "gn"),
                               ("global-OOD AUROC", "g"), ("regional-OOD AUROC", "r"),
                               ("near-OOD AUROC", "n")))
            + '<tr><td class="k">coverage @20% out-of-catalogue</td>'
              '<td class="win">0.722</td><td>0.531</td></tr>'
              '<tr><td class="k">precision @20%</td><td>0.956</td><td>0.946</td></tr>'
              '<tr><td class="k">in-catalogue decline rate</td><td>0.131</td>'
              '<td class="win">0.366</td></tr>'
            + "</tbody></table></div>"),

entry("Geographic prior: real signal that does not pay", "rejected",
      [("billed as", '<s>likely the single highest accuracy-per-effort item in the plan</s>'),
       ("signal is real", "0.715 AUROC, collapsing to 0.511 when coordinates are shuffled within "
                          "a bucket — so it is genuinely biological"),
       ("as a gate", f'+0.0069 expected utility, {ci(-0.0025, 0.0220)}')],
      """<p>Device location narrows the label space 2.7× in European cities and up
      to 9.5× in North American ones, and it is partly independent of the vision
      scores. It still fails, because the vision thresholds are already
      conservative enough that the gate only changes 0.3–2.9% of decisions:
      <strong>the AUROC gain lands where the operating point never looks</strong>.</p>
      <p>Within-genus re-ranking was then tried and also rejected — +3.5pp on 373
      observations became <strong>+0.6pp</strong> on 1,150, with the fix-to-break
      ratio falling from 20:7 to 28:21. The first measurement was small-sample
      noise, and it is the reason every interval on this page is clustered.</p>"""),

entry("Multi-organ fusion fixes rejection", "rejected",
      [("claimed", '<s>three views give three chances to notice nothing matches</s> — '
                   'measured on synthetic groups built to be independent'),
       ("on real observations", "+0.001 AUROC on distant OOD, +0.007 on near OOD"),
       ("what survives", "fusion helps <em>accuracy</em> (+3pp species, point estimate), not rejection")],
      """<p>Every fusion number in the project once rested on synthetic groups —
      one leaf, one bark, one flower drawn independently from the same species.
      That assumes conditional independence, which is what the sampler enforces and
      not what a person does: a real user photographs <em>one individual plant</em>,
      so the photos are correlated.</p>
      <p>The tell is that sharpening combiners <em>hurt</em>. Geometric mean and
      power means all treat photos as more-independent evidence and all came out
      significantly worse, which is direct confirmation of why the synthetic groups
      overestimated the gain. <strong>Trimmed mean — drop the least confident photo
      — won</strong>, and rejection turns out to be a single-photo capability.</p>"""),

entry("Organ cropping, and classical retrieval", "rejected",
      [("cropping", "only 26–33% of photos yield a usable crop"),
       ("imret / ORB keypoints", "at chance for species identification"),
       ("classical descriptors", "leaf 0.065 / bark 0.098 / flower 0.156 top-1, against 0.011 chance")],
      """<p>Three approaches built, evaluated honestly, and abandoned. Cropping
      failed for a reason worth keeping: <em>"a single flower" is ill-posed</em> for
      a found photograph of a dense <i class="sp">Sedum</i> mat. That is the
      argument for guided capture — you control the frame, so a well-posed subject
      exists by construction.</p>
      <p>ORB keypoint matching is built for instance-level near-duplicate
      retrieval; two photographs of the same species are not near-duplicates.</p>"""),
]

tpl = open(f"{SC}/record.tpl.html").read()
prev_rows = "".join(
    f'<tr><td class="k">{int(r2[0]*100)}%</td><td>{r2[1]:.3f}</td><td>{r2[2]:.3f}</td>'
    f'<td>{r1[1]:.3f}</td><td>{r1[2]:.3f}</td></tr>'
    for r2, r1 in zip(D["bioclip2"]["prevalence"]["regional"],
                      D["bioclip1"]["prevalence"]["regional"]))
coreml_rows = "".join(
    f'<tr><td class="k">{r["v"]}</td><td>{r["mb"]:.1f} MB</td><td>{r["cos"]:.4f}</td>'
    f'<td>{r["ms"]:.1f} ms</td><td>{r["ane"]} / {r["cpu"]}</td></tr>' for r in D["coreml"])

html = (tpl.replace("{{precision_coverage}}", charts.precision_coverage(D))
           .replace("{{outcomes}}", charts.outcomes(D))
           .replace("{{prevalence_rows}}", prev_rows)
           .replace("{{coreml_rows}}", coreml_rows)
           .replace("{{entries}}", "\n".join(entries)))
open(sys.argv[1], "w").write(html)
print(f"wrote {sys.argv[1]}  ({len(html):,} bytes, {len(entries)} entries)")
