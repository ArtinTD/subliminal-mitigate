"""Local report rendering for the MASSIVE--medical ratio analysis.

Formatting is intentionally unchanged from the v2 script so downstream paper
artifacts can be checked byte-for-byte during the package migration.
"""
import json

from .ratio import LATEX_LABELS, METHOD_LABELS


def percent_ci(value, latex=False):
    if value["estimate"] is None:
        return "undefined"
    suffix = r"\%" if latex else "%"
    point = f'{100 * value["estimate"]:.2f}{suffix}'
    if value["ci95"] is None:
        return point + " [not reported]"
    lo, hi = value["ci95"]
    return point + f' [{100 * lo:.2f}, {100 * hi:.2f}]'


def change_ci(value):
    lo, hi = value["ci95"]
    return f'{100 * value["estimate"]:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] pp'


def write_reports(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    lines = ["# Fixed-k=4 MASSIVE--medical ratio analysis with baselines", "",
             "All coordinate intervals are absolute marginal 95% percentile-bootstrap intervals. Medical BAD/request retains all 80 requests; abstentions are not labeled SAFE.", "",
             "| Panel | Method | Scope | MASSIVE correct/360: overall accuracy [95% CI] | Medical BAD/answered | BAD/request [95% CI] | Medical answered/80: coverage [95% CI] |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    table = [r"\begin{table*}[t]", r"\centering", r"\caption{Fixed-$k=4$ panel-ratio sensitivity with baseline coverage. Brackets give marginal 95\% bootstrap intervals.}",
             r"\label{tab:mmu-ratio-results-complete}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Panel & Method & MASSIVE accuracy [95\% CI] & Medical \textsc{Bad}/answered & \textsc{Bad}/request [95\% CI] & Medical coverage [95\% CI] \\", r"\midrule"]
    for arm in data["arms"]:
        if arm["method_id"].startswith("direct_"):
            continue
        m, d = arm["massive"], arm["medical"]
        lines.append(f'| {arm["panel_label"]} | {arm["method_label"]} | {arm["comparison_scope"]} | {m["intent_correct_n"]}/360: {percent_ci(m["intent_accuracy"])} | {d["bad_n"]}/{d["accepted_n"]} | {percent_ci(d["bad_rate"])} | {d["accepted_n"]}/80: {percent_ci(d["coverage"])} |')
        dagger = r"$^{\dagger}$" if arm["comparison_scope"] == "historical_contextual" else ""
        label = arm.get("method_latex_label", LATEX_LABELS.get(arm["method_id"], arm["method_label"].replace("_", r"\_")))
        if arm["method_label"] != METHOD_LABELS.get(arm["method_id"], arm["method_label"]) and "method_latex_label" not in arm:
            label = arm["method_label"].replace("_", r"\_")
        table.append(f'{arm["panel_label"]} & {label}{dagger} & {percent_ci(m["intent_accuracy"], True)} & {d["bad_n"]}/{d["accepted_n"]} & {percent_ci(d["bad_rate"], True)} & {percent_ci(d["coverage"], True)} ' + r"\\")
    lines += ["", "## Conditional endpoints and MASSIVE coverage", "",
              "| Panel | Method | MASSIVE answered/360 | Intent accuracy among answers [95% CI] | Medical BAD among answers | Conditional medical interval warning |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm in data["arms"]:
        if arm["method_id"].startswith("direct_"):
            continue
        m, d = arm["massive"], arm["medical"]
        lines.append(f'| {arm["panel_label"]} | {arm["method_label"]} | {m["accepted_n"]}/360 | {percent_ci(m["conditional_intent_accuracy"])} | {d["bad_n"]}/{d["accepted_n"]}: {percent_ci(d["conditional_bad_rate"])} | {d["conditional_bad_rate"]["warning"] or "none"} |')
    lines += ["", "## Paired delta-minus-comparator changes", "", "Negative BAD change means fewer BAD responses; positive coverage change means more answered requests.", "",
              "| Panel | Comparison | Scope | Intent accuracy change [95% CI] | Medical BAD/request change [95% CI] | Medical coverage change [95% CI] |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in data["paired_method_changes"]:
        lines.append(f'| {row["panel_label"]} | {row["comparison"]} | {row["comparison_scope"]} | {change_ci(row["intent_accuracy_change"])} | {change_ci(row["medical_bad_rate_change"])} | {change_ci(row["medical_coverage_change"])} |')
    lines += ["", "## Paired changes across ratios", "", "Positive BAD change means more BAD responses.", "",
              "| Comparison | Method | Scope | Intent accuracy change [95% CI] | Medical BAD/request change [95% CI] | Medical coverage change [95% CI] |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in data["paired_ratio_changes"]:
        lines.append(f'| {row["comparison"]} | {row["method_label"]} | {row["comparison_scope"]} | {change_ci(row["intent_accuracy_change"])} | {change_ci(row["medical_bad_rate_change"])} | {change_ci(row["medical_coverage_change"])} |')
    lines += ["", "## Separate diagnostics and sensitivity", "",
              "| Panel | Method | Answered | Refusal | Response unparseable | Judge parsing failures | Coherent >=50 | BAD-or-unknown/request [95% CI] |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm in data["arms"]:
        d = arm["medical"]
        lines.append(f'| {arm["panel_label"]} | {arm["method_label"]} | {d["accepted_n"]} | {d["refusal_n"]} | {d["response_unparseable_n"]} | {d["judge_parse_failure_n"]} | {d["coherent_ge50_n"]}/{d["accepted_n"]} | {percent_ci(d["bad_or_unknown_sensitivity"])} |')
    lines += ["", "## Bootstrap and interpretation", "", "10,000 replicates; Python random.Random seed 8172026; type-7 percentile intervals. MASSIVE uses common matched row resamples. Medical uses common matched whole-prompt resamples (16 clusters, five requests each), not 80 independent observations. Paired contrasts bootstrap matched differences directly. Coordinate whiskers are marginal intervals, not joint 2D 95% regions. Conditional ratios, when reported, resample both numerator and answered denominator together.", ""]
    lines += ["- " + text for text in data["limitations"]]
    lines += ["", "## Arm-specific provenance notes", ""]
    lines += [f'- {arm["panel_label"]}, {arm["method_label"]} ({arm["comparison_scope"]}): {arm["source_note"]}' for arm in data["arms"] if arm["source_note"]]
    lines += ["", "## Source binding", "", f'{len(data["sources"])} source bindings are hashed in results.json and observations_bound.json. Compact observations contain only outcome codes and neutral IDs, not medical text, gold intents, API identifiers or blind mappings. Original sealed result files and decisions are unchanged. Zero new API or GPU calls.', ""]
    (output_dir / "results.md").write_text("\n".join(lines))
    table += [r"\bottomrule", r"\end{tabular}%", r"}", r"\vspace{3pt}", r"\begin{minipage}{0.99\textwidth}", r"\footnotesize",
              r"\textit{Notes.} $A$/$B$ denote bad-/benign-medical training roles. MASSIVE accuracy is correct/all 360 requests; abstentions count as incorrect. Medical \textsc{Bad}/request retains all 80 requests (16 prompts $\times$ 5 generations); \textsc{Bad}/answered has the returned-output denominator. Coverage denotes non-abstention, not certification of substantive or safe advice. Abstentions are unjudged and are not classified as safe. Intervals resample MASSIVE rows or whole medical prompt clusters (10,000 replicates; seed 8172026; no multiplicity adjustment). A zero-BAD empirical interval is degenerate and does not establish zero risk. $^{\dagger}$Historical balanced A+B Union, not a fresh weighted 2:2 arm. The original $1A{:}3B$ delta-min arm has one refusal and one distinct response-level unparseable judgment, separate from its three \textsc{Bad} responses. Panels are fixed and nested.",
              r"\end{minipage}", r"\end{table*}", ""]
    (output_dir / "results_table.tex").write_text("\n".join(table))
