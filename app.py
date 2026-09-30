"""Streamlit UI.  Run:  streamlit run app.py"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from fh_analyzer import ui_helpers as ui
from fh_analyzer.config import get_settings
from fh_analyzer.family import CATEGORY_HELP, reparse_cached
from fh_analyzer.grounding import CitationCheck, Finding
from fh_analyzer.pipeline import (
    Analysis,
    analyze,
    attach_family,
    fetch_from_uspto,
    load,
    load_local_folder,
    save,
)

st.set_page_config(page_title="File History Analyzer", page_icon="📜", layout="wide")
S = get_settings()

STATUS_STYLE = {
    "verified": ("green", "✔ verified"),
    "wrong_page": ("orange", "↪ wrong page"),
    "partial": ("orange", "≈ partial match"),
    "unverified": ("red", "✘ not found"),
    "bad_reference": ("red", "✘ bad reference"),
    "uncited": ("gray", "no citation given"),
}
RISK_STYLE = {"high": "red", "medium": "orange", "low": "gray"}

st.markdown("""
<style>
ins.add {background:#1a7f3726; color:inherit; text-decoration:underline; padding:0 2px}
del.rm  {background:#cf222e26; color:inherit; padding:0 2px}
mark.hit {background:#f2cc6080; padding:0 1px}
.claimbox {font-family: ui-serif, Georgia, serif; line-height:1.6; padding:.6rem .8rem;
           border:1px solid rgba(128,128,128,.3); border-radius:6px}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers

def badge(check: CitationCheck | None) -> str:
    if check is None:
        return ":gray[no check]"
    color, label = STATUS_STYLE[check.status]
    s = f":{color}-background[{label} · {check.score:.0f}]"
    if check.low_quality_page:
        s += f" :gray-background[low OCR {check.page_ocr_confidence:.0f}%]"
    if check.short_quote:
        s += " :gray-background[short quote]"
    if check.support:
        c = {"supported": "green", "partial": "orange", "unsupported": "red"}[check.support]
        s += f" :{c}-background[judge: {check.support}]"
    return s


def evidence(a: Analysis, f: Finding, ctx: str) -> None:
    chk = a.grounding.checks.get(f.id) if a.grounding else None
    if chk and chk.status == "uncited":
        st.markdown(f"**{f.id}** · {f.doc_label} &nbsp; {badge(chk)}")
        return
    st.markdown(f"**{f.id}** · {f.doc_label} · p.{f.citation.page} &nbsp; {badge(chk)}")
    st.markdown(f"> {html.escape(f.citation.quote)}")
    if chk and chk.status == "wrong_page":
        st.caption(f"Quote found on p.{chk.matched_page}, not p.{f.citation.page}.")
    if chk and chk.support_reason:
        st.caption(f"Judge: {chk.support_reason}")
    if st.button("Open source page", key=f"open-{ctx}-{f.id}"):
        st.session_state["viewer"] = (f.citation.doc_id,
                                      (chk.matched_page if chk and chk.matched_page else
                                       f.citation.page), f.id)
        st.toast("Opened in the Source tab")


def by_kind(a: Analysis, *kinds: str) -> list[Finding]:
    return [f for f in a.findings if f.kind in kinds]


def highlight(text: str, needle: str | None) -> str:
    return ui.highlight(text, needle)


def page_image(a: Analysis, doc_id: str, page: int):
    return ui.page_image(a.pdf_dir, doc_id, page)


@st.cache_data(show_spinner=False)
def saved_label(path: str, mtime: float) -> str:
    """Dropdown label for a saved analysis. `mtime` is only part of the cache key, so a
    re-run analysis gets a fresh label."""
    return ui.analysis_label(path)


# ------------------------------------------------------------------ sidebar / run

with st.sidebar:
    st.header("📜 File History Analyzer")
    mode = st.radio("Source", ["USPTO (Open Data Portal)", "Local PDF folder",
                               "Load saved analysis"])
    judge = family = False
    if mode == "USPTO (Open Data Portal)":
        number = st.text_input("Patent or application number", placeholder="10,123,456")
        judge = st.checkbox("Run LLM support judge", help="Second pass that checks each "
                            "verified quote actually supports the statement. Adds cost.")
        family = st.checkbox("Patent family (EPO OPS)", value=bool(S.epo_ops_key),
                             disabled=not S.epo_ops_key,
                             help="Pull the INPADOC family and every search-report citation "
                             "(X/Y/A) from EPO Open Patent Services, and flag art that was "
                             "never before the US examiner. Needs EPO_OPS_KEY/SECRET in .env.")
        go = st.button("Analyze", type="primary", disabled=not number)
        if not S.uspto_api_key or not S.anthropic_api_key:
            st.warning("Set USPTO_API_KEY and ANTHROPIC_API_KEY in .env")
    elif mode == "Local PDF folder":
        folder = st.text_input("Folder path")
        judge = st.checkbox("Run LLM support judge")
        go = st.button("Analyze", type="primary", disabled=not folder)
    else:
        saved = sorted(S.cache_dir.glob("*/analysis.json")) + sorted(Path("examples").glob(
            "*.json"))
        pick = st.selectbox("Saved analyses", saved,
                            format_func=lambda p: saved_label(str(p), p.stat().st_mtime))
        up = st.file_uploader("…or upload analysis.json", type="json")
        go = st.button("Load", type="primary", disabled=not (pick or up))
    st.caption(f"Model: `{S.model}`")
    if S.is_private:
        st.caption(f"🔒 Private workspace: `{S.workspace}`")

if go:
    try:
        if mode == "Load saved analysis":
            st.session_state["a"] = (Analysis.model_validate_json(up.getvalue()) if up
                                     else load(pick))
        else:
            bar = st.progress(0.0, "Starting…")
            prog = lambda m, f: bar.progress(min(f, 1.0), m)  # noqa: E731
            from fh_analyzer.llm import AnthropicLLM
            if mode.startswith("USPTO"):
                app, docs, texts = fetch_from_uspto(number, S, prog)
                pdf_dir = S.cache_dir / app.application_number / "pdf"
            else:
                app, docs, texts = load_local_folder(Path(folder), S, prog)
                pdf_dir = Path(folder)
            llm = AnthropicLLM(S.anthropic_api_key or "", S.model,
                               cost_log=S.new_cost_log(app.application_number))
            a = analyze(app, docs, texts, llm, S, progress=prog, judge=llm if judge else None,
                        extract_cache=S.cache_dir / app.application_number / "extract")
            a.pdf_dir = str(pdf_dir)
            if family and a.app.patent_number:
                try:
                    attach_family(a, S, prog)
                except Exception as e:  # family is a bonus; never lose the main analysis
                    a.errors["family"] = f"{type(e).__name__}: {e}"
            save(a, S.cache_dir / app.application_number / "analysis.json")
            st.session_state["a"] = a
            bar.empty()
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

a: Analysis | None = st.session_state.get("a")
if a is None:
    st.title("Prosecution history, extracted and verified")
    st.markdown(
        "Enter a US patent number. The app pulls the file wrapper from the USPTO Open Data "
        "Portal, OCRs the substantive documents, has an LLM extract rejections, amendments, "
        + ("estoppel statements, " if S.estoppel else "") +
        "claim versions and reasons for allowance, and then **checks "
        "every cited quote against the OCR text** so you can see what to trust."
    )
    st.stop()

# ------------------------------------------------------------------ header

app = a.app
st.title(app.title or f"Application {app.application_number}")
bib = [f"**Patent** {app.patent_number or '—'}", f"**Application** {app.application_number}",
       f"**Filed** {app.filing_date or '—'}", f"**Granted** {app.grant_date or '—'}",
       f"**Art unit** {app.art_unit or '—'}", f"**Examiner** {app.examiner or '—'}"]
st.markdown(" &nbsp;·&nbsp; ".join(bib))

g = a.grounding
m = st.columns(5)
m[0].metric("Docs analyzed", f"{len(a.extractions)} / {len(a.documents)}")
m[1].metric("Findings", len(a.findings))
m[2].metric("Quotes verified", f"{g.rate_verified:.0%}" if g else "—")
m[3].metric("Pages OCR'd", f"{a.ocr.ocr_pages} / {a.ocr.pages}",
            help=f"Mean OCR confidence {a.ocr.mean_ocr_confidence}")
if a.cost:
    c = a.cost
    m[4].metric("Model cost", c.label,
                help=f"{c.calls} API calls ({c.retries} retries, {c.failed_calls} failed), "
                     f"{c.cached_extractions} documents served from cache at $0; "
                     f"{c.cache_hit_share or 0:.0%} of input read from the prompt cache. "
                     f"Prices as of {c.pricing_as_of}. Details on the Evaluation tab.")
else:
    m[4].metric("LLM calls", a.usage.get("calls", 0),
                help=f"in {a.usage.get('input_tokens', 0):,} / out "
                     f"{a.usage.get('output_tokens', 0):,} tokens (no cost log for this run)")
if a.errors:
    st.warning("Some steps failed: " + "; ".join(f"{k}: {v}" for k, v in a.errors.items()))

TAB_NAMES = ["Overview", "Timeline", "Rejections", "Claim evolution", "Estoppel",
             "Allowance", "Family", "Priority", "Evaluation", "Source"]
if not S.estoppel:              # estoppel analysis is switched off for now (FHA_ESTOPPEL=1)
    TAB_NAMES.remove("Estoppel")
T = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES), strict=True))

# ------------------------------------------------------------------ overview
with T["Overview"]:
    if a.synthesis:
        st.markdown(a.synthesis.overview)
        st.subheader("Key points")
        ids = {f.id: f for f in a.findings}
        for p in a.synthesis.key_points:
            sup = []
            for s in p.support:
                chk = g.checks.get(s) if g else None
                color = STATUS_STYLE[chk.status][0] if chk else "red"
                sup.append(f":{color}-background[{s}]" if s in ids else f":red-background[{s}?]")
            st.markdown(f"- {p.text} &nbsp; " + " ".join(sup))
        st.caption("Badges are the supporting findings, colored by whether their quote was "
                   "verified. A red `?` badge references a finding that does not exist.")
    else:
        st.info("No synthesis produced.")

# ------------------------------------------------------------------ timeline
with T["Timeline"]:
    rows = []
    analyzed = set(a.extractions)
    for d in a.documents:
        rows.append({"date": d.official_date, "code": d.code, "document": d.description,
                     "category": d.category.value, "pages": d.page_count,
                     "analyzed": "✔" if d.doc_id in analyzed else
                                 ("error" if d.doc_id in a.errors else "")})
    show_all = st.toggle("Show all documents (incl. forms, fees, receipts)", value=False)
    df = pd.DataFrame(rows)
    if not show_all:
        df = df[df.category != "other"]
    st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("What happened")
    for f in by_kind(a, "action", "response", "interview", "allowance"):
        with st.expander(f"{f.date} · {f.statement[:140]}"):
            evidence(a, f, "timeline")

# ------------------------------------------------------------------ rejections
with T["Rejections"]:
    rej = by_kind(a, "rejection")
    if not rej:
        st.info("No rejections extracted.")
    else:
        df = pd.DataFrame([{
            "id": f.id, "date": f.date, "claims": ", ".join(map(str, f.meta.get("claims", []))),
            "basis": f.meta.get("basis"), "references": "; ".join(f.meta.get("references", [])),
            "grounding": g.checks[f.id].status if g else "",
        } for f in rej])
        st.dataframe(df, hide_index=True, width="stretch")
        # References applied in the rejections, each linked to Google Patents.
        cited: dict[str, list[str]] = {}
        for f in rej:
            for r in f.meta.get("references", []):
                cited.setdefault(r.strip(), []).append(f.id)
        if cited:
            st.subheader("References discussed")
            lines = []
            for r, ids in sorted(cited.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                url = ui.reference_url(r)
                name = f"[{r}]({url})" if url else f"{r} *(no patent number given)*"
                n = len(ids)
                lines.append(f"- {name} · {n} rejection{'s' if n != 1 else ''} "
                             f"({', '.join(dict.fromkeys(ids))})")
            st.markdown("\n".join(lines))
        for f in rej:
            with st.expander(f"{f.id} · {f.date} · §{f.meta.get('basis')} · claims "
                             f"{f.meta.get('claims')}"):
                st.markdown(ui.linkify_patents(f.statement))
                evidence(a, f, "rejections")
        for f in by_kind(a, "allowable"):
            st.success(f"{f.date}: {f.statement}")

# ------------------------------------------------------------------ claims
with T["Claim evolution"]:
    if not a.claim_histories:
        st.info("No claim listings were extracted.")
    else:
        st.caption("Each claim listing is transcribed by the model; the comparison between "
                   "versions is computed in code (word-level diff), so the changes shown here "
                   "are reproducible. Green = added, red = removed.")
        nums = [h.number for h in a.claim_histories]
        n = st.selectbox("Independent claim", nums,
                         format_func=lambda x: f"Claim {x}" + (
                             " (canceled)" if next(h for h in a.claim_histories
                                                   if h.number == x).canceled else ""))
        h = next(h for h in a.claim_histories if h.number == n)
        for c in h.changes:
            doc = a.doc(c.to_doc)
            title = f"{c.date} · {c.status}" + (f" · {doc.description}" if doc else "")
            if c.noise_suspect:
                title += " · ⚠ likely OCR noise (status says unchanged)"
            with st.expander(title, expanded=not c.noise_suspect):
                st.markdown(f'<div class="claimbox">{c.diff_html}</div>',
                            unsafe_allow_html=True)
                if c.added:
                    st.caption("Added: " + " | ".join(c.added))
        if h.final_text:
            st.subheader("Final text")
            st.markdown(f'<div class="claimbox">{html.escape(h.final_text)}</div>',
                        unsafe_allow_html=True)

    if "Estoppel" not in T:     # amendments normally live on the Estoppel tab
        amend = by_kind(a, "amendment")
        with st.expander(f"All claim amendments ({len(amend)})"):
            for f in amend:
                st.markdown(f"**{f.date}** — {f.statement}")
                evidence(a, f, "amend")

# ------------------------------------------------------------------ estoppel
if "Estoppel" in T:
    with T["Estoppel"]:
        est = by_kind(a, "estoppel")
        order = {"high": 0, "medium": 1, "low": 2}
        est.sort(key=lambda f: (order.get(f.meta.get("risk"), 3), f.date or ""))
        risks = st.multiselect("Risk", ["high", "medium", "low"], default=["high", "medium"])
        amend = by_kind(a, "amendment")
        st.caption(f"{len(est)} potential estoppel statements · {len(amend)} claim amendments")
        for f in [f for f in est if f.meta.get("risk") in risks]:
            r = f.meta.get("risk")
            with st.expander(f":{RISK_STYLE[r]}[{r.upper()}] · {f.date} · claims "
                             f"{f.meta.get('claims')} · {f.meta.get('limitation')}"):
                st.markdown(f"**{f.meta.get('estoppel_type')}** — {f.statement.split(': ', 1)[-1]}")
                if f.meta.get("distinguished_art"):
                    st.markdown("Distinguished: " + ", ".join(f.meta["distinguished_art"]))
                st.caption(f"Why {r}: {f.meta.get('rationale')}")
                evidence(a, f, "estoppel")
        with st.expander(f"All claim amendments ({len(amend)})"):
            for f in amend:
                st.markdown(f"**{f.date}** — {f.statement}")
                evidence(a, f, "amend")

# ------------------------------------------------------------------ allowance
with T["Allowance"]:
    for f in by_kind(a, "allowance"):
        st.markdown(f"**{f.date}** — {f.statement}")
    for f in by_kind(a, "allowance_reason", "examiners_amendment"):
        with st.expander(f"{f.id} · {f.statement[:150]}", expanded=True):
            evidence(a, f, "allowance")
    if not by_kind(a, "allowance", "allowance_reason"):
        st.info("No notice of allowance analyzed.")

# ------------------------------------------------------------------ patent family
with T["Family"]:
    fam = a.family
    if fam is None:
        st.info("No patent-family data for this analysis. EPO Open Patent Services returns "
                "the INPADOC family and every search-report citation with its X/Y/A "
                "category, so you can see art other offices found that the US examiner "
                "never saw.")
        if not S.epo_ops_key:
            st.caption("Set EPO_OPS_KEY and EPO_OPS_SECRET in .env (free registration at "
                       "developers.epo.org) to enable this.")
        elif not a.app.patent_number:
            st.caption("Needs a granted US patent number.")
        elif st.button("Look up family on EPO OPS", type="primary"):
            try:
                with st.spinner("Querying EPO OPS…"):
                    attach_family(a, S)
                save(a, S.cache_dir / a.app.application_number / "analysis.json")
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    else:
        if any(m.biblio_available is None for m in fam.members):
            fam = a.family = reparse_cached(fam, S.cache_dir)   # upgrade older saved data
        subj = ui.fmt_patent(fam.subject.removeprefix("US")) or fam.subject  # "US 8,046,721"
        new = fam.new_art
        xy_new = [c for c in new if c.best_category in ("X", "Y")]
        fm = st.columns(4)
        fm[0].metric("Family members", len(fam.members), help=", ".join(fam.offices))
        fm[1].metric("Distinct citations", len(fam.citations))
        fm[2].metric(f"Not of record in {subj}", len(new))
        fm[3].metric("…graded X or Y", len(xy_new))
        st.caption(
            f"**In record of {subj}** = cited on the face of {subj} (examiner- or "
            "applicant-cited) or named in a rejection / distinguished-art finding from its "
            "file history. Other US family members' citations don't count. Non-patent literature "
            "can't be matched automatically, so it is shown as '?'. Categories: "
            + "; ".join(f"**{k}** {v}" for k, v in list(CATEGORY_HELP.items())[:3]) + "."
        )
        only_new = st.toggle(f"Only art not of record in {subj}", value=True)
        cats = st.multiselect("Category", ["X", "Y", "A", "E", "P", "other", "none"],
                              default=["X", "Y", "E", "P", "other", "none"])

        def _cat_bucket(c):
            b = c.best_category
            return "none" if b is None else b if b in ("X", "Y", "A", "E", "P") else "other"

        rows = []
        for c in fam.citations:
            if only_new and c.of_record_us is not False:
                continue
            if _cat_bucket(c) not in cats:
                continue
            rows.append({
                "cat": "".join(c.categories) or "—",
                "reference": c.display,
                f"in record of {subj}": {True: "✔", False: "✘ new",
                                         None: "?"}[c.of_record_us],
                "cited in": ", ".join(dict.fromkeys(ci.member for ci in c.cited_in)),
                "phase": ", ".join(dict.fromkeys(ci.phase or "" for ci in c.cited_in if
                                                 ci.phase)),
                "claims": "; ".join(ci.rel_claims for ci in c.cited_in if ci.rel_claims),
                "passages": " | ".join(p for ci in c.cited_in for p in ci.passages),
                "link": c.espacenet_url,
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                "link": st.column_config.LinkColumn("Espacenet", display_text="open"),
                "passages": st.column_config.TextColumn(width="large"),
            })
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{fam.subject}_family_citations.csv",
                               mime="text/csv")
        else:
            st.success("Nothing matches the filters.")

        st.subheader("Family members")
        missing = fam.members_without_biblio
        if missing:
            st.warning(
                f"EPO OPS returned no bibliographic record for {len(missing)} of "
                f"{len(fam.members)} members, so their citations are unknown (shown as "
                "blank, not 0). Click **Refresh from EPO OPS** to fetch them.")
        if st.button("🔄 Refresh from EPO OPS", help="Re-read the family and fetch any "
                     "members whose bibliographic data is missing"):
            try:
                with st.spinner("Querying EPO OPS…"):
                    attach_family(a, S, retry_missing=True)
                save(a, S.cache_dir / a.app.application_number / "analysis.json")
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        st.dataframe(pd.DataFrame([{
            "publication": m.label, "date": m.pub_date, "title": m.title,
            "citations": m.citation_count if m.biblio_available else None,
            "note": m.citations_note,
            "family": "simple" if m.simple_family else "extended (INPADOC)",
            "Espacenet": m.espacenet_url, "EP Register": m.register_url,
        } for m in fam.members]), hide_index=True, width="stretch", column_config={
            "Espacenet": st.column_config.LinkColumn(display_text="open"),
            "EP Register": st.column_config.LinkColumn(display_text="documents"),
        })
        st.caption("The EP Register 'documents' link opens the European file wrapper "
                   "(search opinion, examination reports, replies) for EP members.")

# ------------------------------------------------------------------ priority chain
COLOR_ICON = {"green": "🟩", "yellow": "🟨", "red": "🟥"}

with T["Priority"]:
    from fh_analyzer import priority as prio

    st.caption(
        "Is each claim entitled to an earlier filing date in the continuity chain? For every "
        "application (back to the earliest provisional) the specification, claims and "
        "abstract **as filed** are checked for written-description support of each claim "
        "element. A claim gets an application's date only if that application **and every "
        "application between it and this patent** support all of its elements. "
        "🟩 explicit support with a verified quote · 🟨 arguable / implicit, or no verified "
        "quote · 🟥 no support found. An analysis aid, not a legal conclusion.")
    pr = a.priority
    can_run = bool(S.uspto_api_key and S.anthropic_api_key)
    if not can_run:
        st.warning("Needs USPTO_API_KEY and ANTHROPIC_API_KEY in .env.")

    def _run_step(fn):
        try:
            from fh_analyzer.llm import AnthropicLLM
            from fh_analyzer.uspto import ODPClient
            bar = st.progress(0.0, "Starting…")
            with ODPClient(S.uspto_api_key or "", S.cache_dir) as client:
                llm = AnthropicLLM(S.anthropic_api_key or "", S.model,
                                   cost_log=S.new_cost_log(a.app.application_number))
                a.priority = fn(client, llm, lambda m, f: bar.progress(min(f, 1.0), m))
            save(a, S.cache_dir / a.app.application_number / "analysis.json")
            st.rerun()
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    with st.expander("Use pasted claims instead (e.g. the issued claims as numbered in the "
                     "patent)", expanded=bool(pr and pr.errors.get("claims"))):
        pasted = st.text_area("Claims", height=160, key="prio-paste",
                              placeholder="1. A method comprising: ...\n2. The method of "
                                          "claim 1, wherein ...")
        st.caption("By default claims come from the latest claim listing in the file "
                   "history, which uses application numbering; issued patents are often "
                   "renumbered.")
    label1 = "① Find continuity chain & claims" if pr is None else "↻ Re-read chain & claims"
    if st.button(label1, type="primary" if pr is None else "secondary", disabled=not can_run):
        _run_step(lambda c, llm, prog: prio.prepare(c, a, llm, claims_text=pasted or None,
                                                    progress=prog))

    if pr is not None:
        ch = pr.chain
        st.subheader("Continuity chain")
        rel = {}
        for e in ch.edges:
            rel.setdefault(e.child, []).append(
                f"{prio.RELATION_LABEL.get(e.code, e.code or '?')} of {prio.fmt_app(e.parent)}")
        st.dataframe(pd.DataFrame([{
            "application": x.label, "filed": x.filing_date,
            "relationship": "; ".join(rel.get(x.app_no, [])) or "—",
            "status": x.status or ("this patent" if x.app_no == ch.subject else ""),
            "disclosure": (f"{pr.disclosures[x.app_no].pages} pages"
                           if x.app_no in pr.disclosures and pr.disclosures[x.app_no].available
                           else pr.errors.get(x.app_no) or ""),
        } for x in ch.ordered()]), hide_index=True, width="stretch")
        if len(ch.apps) == 1:
            st.info("No parent applications found: the patent's own filing date applies.")
        if pr.errors.get("claims"):
            st.warning(pr.errors["claims"])

        if pr.claims:
            st.subheader("Claims to analyze")
            st.caption(f"Source: {pr.claims_source}")
            indep = [c.number for c in pr.claims if c.depends_on is None]
            dep = [c.number for c in pr.claims if c.depends_on is not None]
            cc = st.columns(2)
            pick_i = cc[0].multiselect("Independent claims", indep,
                                       default=[n for n in pr.selected if n in indep] or indep)
            pick_d = cc[1].multiselect("Dependent claims (optional)", dep,
                                       default=[n for n in pr.selected if n in dep],
                                       format_func=lambda n: f"{n} (from {pr.claim(n).depends_on})")
            with st.expander("Claim text"):
                for c in pr.claims:
                    st.markdown(f"**{c.number}.** {c.text}")
            picked = sorted(pick_i + pick_d)
            n_apps = len(ch.apps)
            if st.button(f"② Analyze support in {n_apps} application(s)", type="primary",
                         disabled=not (picked and can_run)):
                _run_step(lambda c, llm, prog: prio.run(c, pr, llm, S, picked, prog))
            from fh_analyzer.costs import estimate_priority_usd
            known = [sum(len(p.text) for t in d.texts.values() for p in t.pages)
                     for d in pr.disclosures.values() if d.available]
            chars = known + [60_000] * max(0, n_apps - len(known))   # ~25 pages if unknown
            n_el = sum(4 if pr.claim(n).depends_on is None else 1 for n in picked)
            est = estimate_priority_usd(S.model, chars, n_el)
            st.caption((f"Estimated cost ≈ ${est:,.2f} " if est is not None else
                        "No price for this model in pricing.json; ")
                       + f"({n_apps} application(s), ~{n_el} claim elements"
                       + ("" if len(known) == n_apps else
                          ", assuming ~25 pages for specifications not yet downloaded")
                       + "). The specification is cached across batches, so extra "
                       "elements are cheap.")

        if pr.results:
            apps = [x for x in ch.ordered()]
            st.subheader("Earliest supported date per claim")
            st.dataframe(pd.DataFrame([{
                "claim": r.claim,
                "earliest date (all 🟩)": r.strict_date or "—",
                "via": ch.apps[r.strict_app].label if r.strict_app else "—",
                "earliest date (🟩 or 🟨)": r.lenient_date or "—",
                "via ": ch.apps[r.lenient_app].label if r.lenient_app else "—",
                "why not earlier": " | ".join(r.breaks) or "",
            } for r in pr.results]), hide_index=True, width="stretch")
            st.caption("If a claim shows '—', not every element was found even in this "
                       "patent's own application as filed - check those elements by hand.")

            grid = pr.grid()
            rows = []
            for el in pr.elements:
                row = {"element": el.id, "text": el.text}
                for x in apps:
                    c = grid.get((x.app_no, el.id))
                    row[x.label] = (COLOR_ICON[c.color] + (" ✱" if c.new_matter else "")) \
                        if c else "·"
                rows.append(row)
            st.subheader("Support grid (oldest application → this patent)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("✱ = support found only in text a CIP added over its parent. "
                       "· = application not analyzed (see chain table).")

            eid = st.selectbox("Show evidence for element", [e.id for e in pr.elements],
                               format_func=lambda i: f"{i}: "
                               + next(e.text for e in pr.elements if e.id == i)[:90])
            for x in apps:
                c = grid.get((x.app_no, eid))
                if c is None:
                    continue
                with st.expander(f"{COLOR_ICON[c.color]} {x.label} · filed {x.filing_date} · "
                                 f"{c.level}" + (" · CIP new matter" if c.new_matter else ""),
                                 expanded=c.color != "red"):
                    st.write(c.explanation)
                    disc = pr.disclosures.get(x.app_no)
                    for q, ok in zip(c.citations, c.verified, strict=False):
                        part = disc.parts.get(q.doc_id, "") if disc else ""
                        st.markdown(f"{'✔' if ok else '✘ not found in OCR text'} · {part} "
                                    f"p.{q.page}\n> {html.escape(q.quote)}")

            for nm in pr.new_matter:
                with st.expander(f"CIP new matter: {prio.fmt_app(nm.child)} vs parent "
                                 f"{prio.fmt_app(nm.parent)} — {len(nm.new_passages)} new "
                                 f"passage(s), ~{nm.new_fraction:.0%} of sentences"):
                    for page, t in nm.new_passages:
                        st.markdown(f"**p.{page}** {html.escape(t)}")
            if pr.element_warnings:
                st.warning("Element split check: " + "; ".join(pr.element_warnings))
            other = {k: v for k, v in pr.errors.items() if k != "claims"}
            if other:
                st.warning("Some applications could not be analyzed: "
                           + "; ".join(f"{prio.fmt_app(k)}: {v}" for k, v in other.items()))
            if pr.costs:
                last = pr.costs[-1]
                st.caption(f"Cost of this priority analysis: ${pr.total_usd:,.2f} over "
                           f"{len(pr.costs)} step(s); last step {last.label}, {last.calls} "
                           f"calls, {last.cache_hit_share or 0:.0%} prompt-cache hits.")

# ------------------------------------------------------------------ evaluation
with T["Evaluation"]:
    if not g:
        st.info("No grounding report.")
    else:
        st.markdown(
            "Every finding carries a verbatim quote and a page pin cite. The quote is fuzzy-"
            "matched (rapidfuzz, threshold 90) against the OCR text of the cited page; if it "
            "is not there, the rest of the document is searched to separate a **wrong pin "
            "cite** from an **invented quote**. Optionally, a second LLM pass judges whether "
            "the located text actually *supports* the statement."
        )
        c = st.columns(4)
        c[0].metric("Verified on cited page", f"{g.rate_verified:.0%}")
        c[1].metric("Text found anywhere", f"{g.rate_grounded:.0%}")
        c[2].metric("Not found / uncited", g.totals.get("unverified", 0)
                    + g.totals.get("bad_reference", 0) + g.totals.get("uncited", 0))
        c[3].metric("Failures on low-OCR pages",
                    "—" if g.low_quality_share_of_failures is None
                    else f"{g.low_quality_share_of_failures:.0%}",
                    help="Share of partial/unverified quotes on pages with OCR confidence "
                         f"< {S.low_conf_threshold:.0f}%. High = OCR problem, low = model "
                         "problem.")
        left, right = st.columns(2)
        with left:
            st.subheader("By finding type")
            st.dataframe(pd.DataFrame(g.by_kind).T.fillna(0).astype(int),
                         width="stretch")
        with right:
            st.subheader("By document")
            st.dataframe(pd.DataFrame(g.by_doc).T.fillna(0).astype(int),
                         width="stretch")
        if g.support_totals:
            st.subheader("LLM judge: does the quote support the statement?")
            st.bar_chart(pd.Series(g.support_totals))
        if g.synthesis:
            st.subheader("Overview traceability")
            s = g.synthesis
            st.write(f"{s['points']} key points · {s['points_anchored_to_verified_quote']} "
                     f"anchored to a verified quote · {s['points_without_valid_support']} with no "
                     f"valid support · dangling ids: {s['dangling_ids'] or 'none'}")
        st.subheader("Completeness tripwires")
        st.caption("Grounding catches wrong evidence; it cannot catch omissions. These simple "
                   "checks on the OCR text flag documents where the model seems to have "
                   "missed content (e.g. an action that says 'rejected under' but no "
                   "rejections recorded). Each tripped check gets one targeted retry; only "
                   "issues that survive the retry are listed here.")
        if a.completeness:
            for did, issues in a.completeness.items():
                d = a.doc(did)
                st.warning(f"**{d.label if d else did}** — " + " ".join(issues))
        else:
            st.success("No unresolved omissions detected.")
        st.subheader("Review queue")
        st.caption("Items a human should check first: not found, wrong page, partial, "
                   "judged unsupported.")
        bad = [f for f in a.findings
               if g.checks[f.id].status != "verified"
               or g.checks[f.id].support in ("unsupported", "partial")]
        for f in bad:
            with st.expander(f"{f.id} · {f.kind} · {f.statement[:120]}"):
                evidence(a, f, "review")
        if not bad:
            st.success("Nothing to review: all quotes verified.")
        st.subheader("Cost")
        if a.cost is None:
            st.caption("This analysis was run before cost accounting was added.")
        else:
            c = a.cost
            cc = st.columns(4)
            cc[0].metric("Total", c.label)
            cc[1].metric("API calls", c.calls, help=f"{c.retries} retries, "
                         f"{c.failed_calls} failed; {c.cached_extractions} documents came "
                         "from the extraction cache at no cost")
            cc[2].metric("Prompt-cache hits", f"{c.cache_hit_share or 0:.0%}",
                         help="Share of input tokens read from Anthropic's prompt cache "
                              "(billed at a fraction of the normal input price)")
            cc[3].metric("Latency p50 / p95", f"{(c.latency_p50_ms or 0) / 1000:.1f}s / "
                         f"{(c.latency_p95_ms or 0) / 1000:.1f}s")
            left, right = st.columns(2)
            with left:
                st.caption("By stage")
                st.dataframe(pd.Series(c.by_stage, name="USD"), width="stretch")
            with right:
                st.caption("By document type")
                st.dataframe(pd.Series(c.by_doc_code, name="USD"), width="stretch")
            if c.unpriced_models:
                st.warning(f"No price for {', '.join(c.unpriced_models)} in pricing.json, "
                           "so those calls are not in the total.")
            st.caption(f"Prices as of {c.pricing_as_of} (src/fh_analyzer/pricing.json). "
                       "Every call is logged to data/costs/calls.jsonl; run `fh-costs` for "
                       "reports across runs.")
        with st.expander("Run metadata"):
            st.json({"model": a.model, "prompt_version": a.prompt_version,
                     "created": a.created, "usage": a.usage, "ocr": a.ocr.model_dump()})

# ------------------------------------------------------------------ source viewer
with T["Source"]:
    doc_ids = [d for d in a.analyzed_doc_ids if d in a.texts]
    if not doc_ids:
        st.info("No document text stored.")
    else:
        default = st.session_state.get("viewer", (doc_ids[0], 1, None))
        di = doc_ids.index(default[0]) if default[0] in doc_ids else 0
        d_id = st.selectbox("Document", doc_ids, index=di,
                            format_func=lambda x: a.doc(x).label if a.doc(x) else x)
        dt = a.texts[d_id]
        pg = st.number_input("Page", 1, len(dt.pages),
                             min(default[1], len(dt.pages)) if default[0] == d_id else 1)
        page = dt.page(pg)
        needle = None
        if default[2] and default[0] == d_id and g:
            chk = g.checks.get(default[2])
            needle = chk.matched_text if chk else None
        st.caption(f"{page.method}" + (f" · OCR confidence {page.confidence}%"
                                        if page.confidence is not None else ""))
        img = page_image(a, d_id, pg)
        left_col, r = st.columns([1, 1]) if img else (st.container(), None)
        with left_col:
            st.markdown(f'<div class="claimbox" style="max-height:70vh;overflow:auto">'
                        f"{highlight(page.text, needle)}</div>", unsafe_allow_html=True)
        if img and r:
            with r:
                st.image(img, caption="Original page image")
