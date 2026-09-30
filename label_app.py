"""Gold-label review tool.  Run:  python -m streamlit run label_app.py

Pre-fills labels from a saved analysis. For each document, confirm, fix, or reject
each model finding, add anything the model missed, then mark the document reviewed.
Every click is saved to gold/<application>.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from fh_analyzer import ui_helpers as ui
from fh_analyzer.config import get_settings
from fh_analyzer.evalset import (
    CHOICES,
    ERROR_TAGS,
    KIND_FIELDS,
    KIND_LABELS,
    GoldItem,
    GoldSet,
    fmt_claims,
    gold_from_analysis,
    gold_path,
    load_gold,
    parse_claims,
    save_gold,
    verdict_stats,
)
from fh_analyzer.pipeline import Analysis, load
from fh_analyzer.prompts import PROMPT_VERSION

st.set_page_config(page_title="Gold Label Review", page_icon="🏷️", layout="wide")
S = get_settings()
GOLD_DIR = S.gold_dir

st.markdown("""
<style>
mark.hit {background:#f2cc6099; padding:0 1px}
.src {font-family: ui-serif, Georgia, serif; line-height:1.55; font-size:.92rem;
      padding:.6rem .8rem; border:1px solid rgba(128,128,128,.3); border-radius:6px;
      max-height:72vh; overflow:auto}
.quote {font-size:.85rem; opacity:.8; font-style:italic}
div[data-testid="stVerticalBlockBorderWrapper"] {margin-bottom:.25rem}
</style>""", unsafe_allow_html=True)

VERDICT_ICON = {None: "⬜", "correct": "✅", "corrected": "✏️", "incorrect": "❌"}
RISK_COLOR = {"high": "red", "medium": "orange", "low": "gray"}


# ------------------------------------------------------------------ data

@st.cache_resource(show_spinner="Loading analysis…")
def load_analysis(path: str, mtime: float) -> Analysis:
    return load(Path(path))


@st.cache_data(show_spinner=False)
def label_for(path: str, mtime: float) -> str:
    return ui.analysis_label(path)


def save() -> None:
    save_gold(st.session_state.gold, st.session_state.gold_file)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ sidebar

runs = sorted(S.cache_dir.glob("*/analysis.json"))
with st.sidebar:
    st.header("🏷️ Gold label review")
    if not runs:
        st.warning("No saved analyses yet. Analyze a patent in the main app first.")
        st.stop()
    # Default to a patent already being labeled, else the first one.
    started = [i for i, r in enumerate(runs) if gold_path(GOLD_DIR, r.parent.name).exists()]
    path = st.selectbox("Patent", runs, index=started[0] if started else 0,
                        format_func=lambda p: ("🏷️ " if gold_path(GOLD_DIR, p.parent.name)
                                               .exists() else "") + label_for(str(p),
                                                                              p.stat().st_mtime))
    labeler = st.text_input("Your name / initials", value=st.session_state.get("labeler", ""))
    st.session_state.labeler = labeler

a = load_analysis(str(path), path.stat().st_mtime)
gfile = gold_path(GOLD_DIR, a.app.application_number)

# (Re)load the gold set when the patent changes.
if st.session_state.get("gold_file") != gfile:
    g = load_gold(gfile)
    st.session_state.gold = g
    st.session_state.gold_file = gfile
    st.session_state.pop("doc", None)
    st.session_state.pop("editing", None)
g: GoldSet | None = st.session_state.gold

with st.sidebar:
    if g is None:
        st.info("No labels for this patent yet. Starting creates a label file pre-filled "
                "with the model's findings for you to check.")
        if a.prompt_version != PROMPT_VERSION:
            st.warning(f"This analysis used prompt {a.prompt_version}; the current prompt "
                       f"is {PROMPT_VERSION}. Re-run it in the main app first for better "
                       "pre-filled labels.")
        start = st.button("Start labeling", type="primary", width="stretch")
if g is None:
    n_items = sum(f.kind in KIND_FIELDS for f in a.findings)
    st.title(ui.analysis_label(str(path)))
    st.markdown(
        f"**{len(a.extractions)} documents** and **{n_items} model findings** to check.\n\n"
        "Labeling turns this patent into part of an expert-labeled benchmark set. "
        "Each model finding is pre-filled; you confirm it, fix it, or reject it, and "
        "add anything the model missed. Click **Start labeling** in the sidebar.")
    if start:
        g = gold_from_analysis(a)
        g.labeler = labeler
        st.session_state.gold = g
        save()
        st.rerun()
    st.stop()
with st.sidebar:
    p = g.progress()
    st.progress(p["docs_reviewed"] / max(p["docs_total"], 1),
                f"Documents reviewed: {p['docs_reviewed']} / {p['docs_total']}")
    st.progress(p["items_reviewed"] / max(p["items_total"], 1),
                f"Model items checked: {p['items_reviewed']} / {p['items_total']}")
    st.caption(f"Items you added (model misses): {p['added']}")
    vs = verdict_stats(g)
    if vs["verdicts"]:
        st.caption("Verdicts: " + " · ".join(f"{VERDICT_ICON[k]} {v}"
                                             for k, v in vs["verdicts"].items()))
    if g.source_prompt_version != PROMPT_VERSION:
        st.caption(f"⚠ Labels were pre-filled from prompt {g.source_prompt_version}.")
    st.caption(f"Saved to `{gfile.as_posix()}` after every click.")
    with st.expander("How to label"):
        st.markdown(
            "1. Pick a document. Its source text is on the right.\n"
            "2. For each item: **✅ Correct**, **✏️ Fix** (edit the fields to the truth), or "
            "**❌ Wrong** (not a real item). Tag *why* when it's wrong or fixed.\n"
            "3. Read the document and **➕ add anything the model missed**. This is what "
            "makes recall measurable.\n"
            "4. Click **Mark document reviewed**. Only reviewed documents are scored.\n\n"
            "Then run `fh-score` to get precision / recall per type.")

# ------------------------------------------------------------------ document navigator

docs = [d for d in a.documents if d.doc_id in g.docs]
if not docs:
    st.warning("This analysis has no analyzed documents.")
    st.stop()


def doc_title(d) -> str:
    items = g.for_doc(d.doc_id)
    pending = sum(i.origin == "model" and i.verdict is None for i in items)
    mark = "✅" if g.docs[d.doc_id].reviewed else ("🟡" if pending < len(items) else "⬜")
    return f"{mark} {d.label}  ·  {len(items)} item(s)"


ids = [d.doc_id for d in docs]
if "_goto" in st.session_state:          # set by buttons below; applied before the widget
    st.session_state.doc = st.session_state.pop("_goto")
if st.session_state.get("doc") not in ids:
    st.session_state.doc = next((i for i in ids if not g.docs[i].reviewed), ids[0])

nav = st.columns([8, 1, 1, 2])
doc_id = nav[0].selectbox("Document", ids, key="doc", label_visibility="collapsed",
                          format_func=lambda i: doc_title(next(d for d in docs if d.doc_id == i)))
cur = ids.index(doc_id)
if nav[1].button("◀", width="stretch", disabled=cur == 0, help="Previous document"):
    st.session_state._goto = ids[cur - 1]
    st.rerun()
if nav[2].button("▶", width="stretch", disabled=cur == len(ids) - 1,
                 help="Next document"):
    st.session_state._goto = ids[cur + 1]
    st.rerun()
nxt = next((i for i in ids[cur + 1:] + ids[:cur] if not g.docs[i].reviewed), None)
if nav[3].button("Next unreviewed ⏭", width="stretch", disabled=nxt is None):
    st.session_state._goto = nxt
    st.rerun()

doc = next(d for d in docs if d.doc_id == doc_id)
gdoc = g.docs[doc_id]
if gdoc.opened_at is None:
    gdoc.opened_at = now()
    save()
dtext = a.texts.get(doc_id)
n_pages = len(dtext.pages) if dtext else 0
if st.session_state.get("viewer_doc") != doc_id:
    st.session_state.viewer_doc = doc_id
    first = min((i for i in g.for_doc(doc_id) if i.page), key=lambda i: i.page, default=None)
    st.session_state.page = first.page if first and first.page <= n_pages else 1
    st.session_state.focus = first.id if first else None

left, right = st.columns([5, 6], gap="large")


# ------------------------------------------------------------------ field widgets

def field_widgets(kind: str, values: dict, key: str) -> dict:
    """Render editable widgets for an item's fields; return the edited values."""
    out = {}
    for name, typ in KIND_FIELDS[kind]:
        v = values.get(name)
        k = f"{key}-{name}"
        label = name.replace("_", " ").capitalize()
        if typ == "claims":
            out[name] = parse_claims(st.text_input(label, fmt_claims(v or []), key=k,
                                                   help="e.g. 1-5, 8"))
        elif typ in CHOICES or typ in ("basis", "etype", "risk", "cstatus"):
            opts = CHOICES[typ]
            idx = opts.index(v) if v in opts else 0
            if typ == "risk":
                out[name] = st.radio(label, opts, index=idx, key=k, horizontal=True)
            else:
                out[name] = st.selectbox(label, opts, index=idx, key=k)
        elif typ == "list":
            raw = st.text_input(label + " (separate with ;)", "; ".join(v or []), key=k)
            out[name] = [x.strip() for x in raw.split(";") if x.strip()]
        elif typ == "str":
            out[name] = st.text_input(label, v or "", key=k)
        else:
            out[name] = st.text_area(label, v or "", key=k,
                                     height=180 if name == "text" else 80)
    return out


def fields_line(kind: str, f: dict) -> str:
    bits = []
    if f.get("claims"):
        bits.append(f"**Claims {fmt_claims(f['claims'])}**")
    if kind == "rejection":
        bits.append(f"§{f.get('basis')}")
        if f.get("references"):
            bits.append("over " + "; ".join(f["references"]))
    if kind == "estoppel":
        r = f.get("risk", "low")
        bits.append(f":{RISK_COLOR.get(r, 'gray')}-background[{r.upper()}]")
        bits.append(f"*{f.get('estoppel_type', '')}*")
        if f.get("limitation"):
            bits.append(f"limitation: “{f['limitation']}”")
    if kind == "claim":
        bits.append(f"*{f.get('status', '')}*")
    return " · ".join(bits)


# ------------------------------------------------------------------ item cards

def set_verdict(it: GoldItem, verdict) -> None:
    it.verdict = verdict
    if verdict == "correct":
        it.fields = dict(it.model_values)
        it.error_tags = []
    save()


def item_card(it: GoldItem) -> None:
    editing = st.session_state.get("editing") == it.id
    with st.container(border=True):
        head = st.columns([7, 2])
        origin = " · ➕ added by you" if it.origin == "added" else ""
        head[0].markdown(f"{VERDICT_ICON[it.verdict]} **{KIND_LABELS.get(it.kind, it.kind)}**"
                         f"{origin}  \n{fields_line(it.kind, it.fields)}")
        if it.page and head[1].button(f"📄 p.{it.page}", key=f"pg-{it.id}",
                                      width="stretch", help="Show in source pane"):
            st.session_state.page = it.page
            st.session_state.focus = it.id
            st.rerun()
        if it.origin == "model" and it.verdict and head[1].button(
                "↺ Undo", key=f"undo-{it.id}", width="stretch", type="tertiary"):
            it.verdict, it.fields, it.error_tags = None, dict(it.model_values), []
            save()
            st.rerun()

        if it.verdict == "correct" and not editing:
            return  # confirmed items collapse to their header line to keep the list short
        summ = it.fields.get("summary") or it.fields.get("change") or it.fields.get("text")
        if summ:
            st.markdown(summ if len(summ) < 600 else summ[:600] + "…")
        if it.quote:
            st.markdown(f'<div class="quote">“{it.quote}”</div>', unsafe_allow_html=True)
        if it.verdict in ("corrected", "incorrect") and (it.error_tags or it.note):
            st.caption("Why: " + ", ".join(it.error_tags) + (f" — {it.note}" if it.note else ""))

        if editing:
            with st.form(f"form-{it.id}", border=False):
                wrong = st.session_state.get("edit_mode") == "wrong"
                vals = it.fields if wrong else field_widgets(it.kind, it.fields, f"e-{it.id}")
                tags = st.multiselect("What was wrong?", ERROR_TAGS, default=it.error_tags,
                                      key=f"tags-{it.id}")
                note = st.text_input("Note (optional)", it.note, key=f"note-{it.id}")
                c = st.columns(2)
                if c[0].form_submit_button("Save", type="primary", width="stretch"):
                    it.fields = dict(vals)
                    it.error_tags, it.note = tags, note
                    if it.origin == "model":
                        it.verdict = "incorrect" if wrong else (
                            "correct" if vals == it.model_values else "corrected")
                    st.session_state.editing = None
                    save()
                    st.rerun()
                if c[1].form_submit_button("Cancel", width="stretch"):
                    st.session_state.editing = None
                    st.rerun()
            return

        b = st.columns(3)
        if it.origin == "model":
            if b[0].button("✅ Correct", key=f"ok-{it.id}", width="stretch",
                           type="primary" if it.verdict is None else "secondary"):
                set_verdict(it, "correct")
                st.rerun()
            if b[1].button("✏️ Fix", key=f"fix-{it.id}", width="stretch"):
                st.session_state.editing, st.session_state.edit_mode = it.id, "fix"
                st.rerun()
            if b[2].button("❌ Wrong", key=f"bad-{it.id}", width="stretch"):
                set_verdict(it, "incorrect")
                st.session_state.editing, st.session_state.edit_mode = it.id, "wrong"
                st.rerun()

        else:
            if b[0].button("✏️ Edit", key=f"fix-{it.id}", width="stretch"):
                st.session_state.editing, st.session_state.edit_mode = it.id, "fix"
                st.rerun()
            if b[1].button("🗑 Delete", key=f"del-{it.id}", width="stretch"):
                g.items.remove(it)
                save()
                st.rerun()


# ------------------------------------------------------------------ left: items

with left:
    items = g.for_doc(doc_id)
    pending = [i for i in items if i.origin == "model" and i.verdict is None]
    st.subheader(doc.description)
    st.caption(f"{doc.official_date} · {doc.code} · {n_pages} page(s) · "
               f"{len(items)} item(s), {len(pending)} unchecked")
    if a.completeness.get(doc_id):
        st.warning("Completeness check flagged this document: "
                   + " ".join(a.completeness[doc_id]))

    if not items:
        st.info("The model recorded nothing to label here. If the document does contain "
                "rejections, amendments, arguments or reasons, add them below. Otherwise "
                "just mark it reviewed.")
    order = {None: 0, "corrected": 1, "incorrect": 2, "correct": 3}
    # Items scroll inside their own box so the source pane stays alongside them.
    with st.container(height=640 if len(items) > 2 else "content", border=False):
        for it in sorted(items, key=lambda i: (order[i.verdict] if i.origin == "model"
                                               else 1, i.page or 0)):
            item_card(it)

    if pending and st.button(f"✅ Mark all {len(pending)} unchecked item(s) correct",
                             width="stretch"):
        for it in pending:
            set_verdict(it, "correct")
        st.rerun()

    with st.expander("➕ Add an item the model missed",
                     expanded=bool(st.session_state.get("adding"))):
        kind = st.selectbox("Type", [k for k in KIND_FIELDS if S.estoppel or k != "estoppel"],
                            format_func=KIND_LABELS.get,
                            key="add-kind")
        with st.form("add-form", clear_on_submit=True):
            vals = field_widgets(kind, {}, f"add-{kind}")
            pc = st.columns([1, 3])
            page = pc[0].number_input("Page", 1, max(n_pages, 1),
                                      st.session_state.get("page", 1), key="add-page")
            quote = pc[1].text_input("Supporting quote (copy from the source pane)",
                                     key="add-quote")
            note = st.text_input("Note (optional)", key="add-note")
            if st.form_submit_button("Add item", type="primary"):
                g.items.append(GoldItem(id=g.next_id(), origin="added", kind=kind,
                                        doc_id=doc_id, fields=vals, page=int(page),
                                        quote=quote.strip(), note=note))
                save()
                st.toast("Added")
                st.rerun()

    st.divider()
    dnote = st.text_input("Document note (optional)", gdoc.note, key=f"dnote-{doc_id}")
    if dnote != gdoc.note:
        gdoc.note = dnote
        save()
    if gdoc.reviewed:
        st.success(f"Reviewed {gdoc.reviewed_at or ''}")
        if st.button("Reopen document"):
            gdoc.reviewed, gdoc.reviewed_at = False, None
            save()
            st.rerun()
    else:
        if pending:
            st.caption(f"{len(pending)} item(s) still unchecked. They must be checked "
                       "before the document can be marked reviewed.")
        if st.button("✔ Mark document reviewed" + (" → next" if nxt else ""),
                     type="primary", width="stretch", disabled=bool(pending)):
            gdoc.reviewed, gdoc.reviewed_at = True, now()
            save()
            if nxt:
                st.session_state._goto = nxt
            st.rerun()

# ------------------------------------------------------------------ right: source pane

with right:
    if not dtext or not n_pages:
        st.info("No text stored for this document.")
    else:
        pc = st.columns([1, 1, 3, 3])
        if pc[0].button("◀", key="pprev", disabled=st.session_state.page <= 1,
                        width="stretch"):
            st.session_state.page -= 1
            st.session_state.focus = None
            st.rerun()
        if pc[1].button("▶", key="pnext", disabled=st.session_state.page >= n_pages,
                        width="stretch"):
            st.session_state.page += 1
            st.session_state.focus = None
            st.rerun()
        pc[2].markdown(f"**Page {st.session_state.page} / {n_pages}**")
        pg = dtext.page(st.session_state.page)
        if pg.confidence is not None:
            pc[3].caption(f"OCR confidence {pg.confidence:.0f}%"
                          + (" ⚠ low" if pg.confidence < S.low_conf_threshold else ""))
        focus = next((i for i in g.items if i.id == st.session_state.get("focus")), None)
        needle = focus.quote if focus and focus.page == st.session_state.page else None
        img = ui.page_image(a.pdf_dir, doc_id, st.session_state.page)
        tabs = st.tabs(["OCR text", "Page image"] if img else ["OCR text"])
        with tabs[0]:
            st.markdown(f'<div class="src">{ui.highlight(pg.text, needle)}</div>',
                        unsafe_allow_html=True)
        if img:
            with tabs[1]:
                st.image(img, width="stretch")
