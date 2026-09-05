"""`plantid plan | build | card`.

Usage:
    PYTHONPATH=. .venv/bin/python -m plantid.tool.cli plan --species my.txt --budget 20
    PYTHONPATH=. .venv/bin/python -m plantid.tool.cli build --species my.txt --out models/my
    PYTHONPATH=. .venv/bin/python -m plantid.tool.cli card models/my
"""

import argparse
import sys
from pathlib import Path

from plantid.tool import build as B
from plantid.tool import card as C
from plantid.tool import encoders, plan as P, species as S


def _species_arg(args) -> list[str]:
    if args.species:
        return S.read_list(args.species)
    if args.name:
        out = []
        for n in args.name:
            c = S.canonical(n)
            if not c:
                raise ValueError(f"could not parse {n!r} as 'Genus species'")
            out.append(c)
        return list(dict.fromkeys(out))
    raise ValueError("give --species FILE or one or more --name 'Genus species'")


def cmd_plan(args):
    chosen = _species_arg(args)
    pl = P.make_plan(chosen, budget_mb=args.budget, p_ood=args.ood_rate,
                     encoder=args.encoder)
    print()
    print(P.render(pl))
    print()
    print(f"  Next: plantid build --species {args.species or '<list>'} "
          f"--encoder {pl['encoder'].variant}")
    print()
    return 0


def cmd_build(args):
    chosen = _species_arg(args)
    comp = S.analyse(chosen)
    enc = (encoders.BY_VARIANT[args.encoder] if args.encoder
           else encoders.choose(args.budget))

    print(f"encoder {enc.label} ({enc.size_mb():.1f} MB int4), "
          f"{len(chosen)} species", file=sys.stderr)
    ds = B.load_local(enc.variant, chosen)
    missing = set(chosen) - set(ds.y_train)
    if missing:
        print(f"warning: no training rows for {len(missing)} species: "
              f"{', '.join(sorted(missing)[:6])}", file=sys.stderr)
    if ds.counts["in_catalog"] == 0:
        raise SystemExit("no evaluation rows for any chosen species -- nothing to measure")
    print(f"  train {ds.counts['train']} rows | eval in-list {ds.counts['in_catalog']}, "
          f"relatives {ds.counts['near_ood']}, unrelated {ds.counts['distant_ood']}",
          file=sys.stderr)

    clf = B.fit_head(ds)
    frame = B.score_frame(clf, ds)
    metrics = B.fit_and_measure(frame, p_ood=args.ood_rate)

    out = B.save_bundle(Path(args.out), clf, chosen, enc.variant, metrics, comp,
                        ds.counts, source="local-catalogue")
    card_path = C.write(out)
    print(f"\nbundle {out}\ncard   {card_path}", file=sys.stderr)
    print(f"\n  coverage {100*metrics['coverage']:.1f}%  "
          f"precision {100*metrics['precision']:.1f}%  "
          f"species-level {100*metrics['species_share']:.1f}%")
    return 0


def cmd_card(args):
    manifest = B.load_bundle(Path(args.bundle))
    print(C.render(manifest))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="plantid", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=False):
        p.add_argument("--species", help="file with one 'Genus species' per line")
        p.add_argument("--name", action="append", help="a species; repeatable")
        p.add_argument("--budget", type=float, metavar="MB",
                       help="size budget for the encoder, in MB")
        p.add_argument("--encoder", choices=sorted(encoders.BY_VARIANT),
                       help="override the budget-based choice")
        p.add_argument("--ood-rate", type=float, default=0.2, metavar="P",
                       help="assumed share of queries not on your list (default 0.2)")
        if out:
            p.add_argument("--out", required=True, help="bundle directory to write")

    p_plan = sub.add_parser("plan", help="what this species list will give you")
    common(p_plan)
    p_plan.set_defaults(fn=cmd_plan)

    p_build = sub.add_parser("build", help="fit, measure, and write a bundle")
    common(p_build, out=True)
    p_build.set_defaults(fn=cmd_build)

    p_card = sub.add_parser("card", help="print the card for a built bundle")
    p_card.add_argument("bundle")
    p_card.set_defaults(fn=cmd_card)

    args = ap.parse_args(argv)
    if args.cmd == "plan" and args.ood_rate not in P.projection.measured_p_ood():
        ap.error(f"--ood-rate for `plan` must be one of "
                 f"{P.projection.measured_p_ood()} (the rates measured); "
                 f"`build` accepts any value because it fits on your data")
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
