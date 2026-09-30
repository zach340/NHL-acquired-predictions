"""
components.py
=============
Reusable Streamlit building blocks shared by the tabs.
"""

import uuid

import streamlit as st
import streamlit.components.v1 as components

from .. import nhl_api
from ..config import get_team_color
from ..grading import score_to_grade

NHL_LOGO_URL = "https://assets.nhle.com/logos/nhl/svg/{team}_light.svg"


def pct_color(pct):
    return ("#FFD700" if pct >= 90 else "#4a90d9" if pct >= 75 else
            "#57a85a" if pct >= 50 else "#e8a838" if pct >= 35 else "#c8102e")


def pct_bar(label, value, pct, lower_is_better=False):
    """Label, value and coloured percentile text over a progress bar."""
    if lower_is_better:
        pct_label = f"Top {100 - pct:.0f}%" if pct <= 90 else "Elite"
        note = "<span style='color:#888;font-size:0.8em'>↓ lower is better</span>"
    else:
        pct_label, note = f"{pct:.0f}th%", ""
    st.markdown(
        f"**{label}** &nbsp; `{value}` &nbsp; — &nbsp; "
        f"<span style='color:{pct_color(pct)}'>**{pct_label}**</span> {note}",
        unsafe_allow_html=True,
    )
    st.progress(int(pct))


def grade_metrics(items):
    """Row of st.metric grade cards from [(label, grade, pct_or_None)]."""
    for col, (label, grade, pct) in zip(st.columns(len(items)), items):
        col.metric(label, grade, f"Top {100 - pct:.0f}%" if pct is not None else "")


def grade_item(label, pct):
    return label, score_to_grade(pct), pct


def headshot_html(player_id, size=80):
    """Circular headshot (base64, avoids CDN blocking) or a silhouette SVG."""
    b64 = nhl_api.fetch_headshot_b64(player_id)
    if b64:
        style = (f"width:{size}px;height:{size}px;border-radius:50%;"
                 "object-fit:cover;margin-top:4px;background:#1a1a2e")
        return f'<img src="data:image/png;base64,{b64}" style="{style}">'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="50" fill="#2d3748"/>'
        '<circle cx="50" cy="38" r="18" fill="#718096"/>'
        '<ellipse cx="50" cy="80" rx="28" ry="20" fill="#718096"/>'
        '</svg>'
    )


def player_header(player_id, title, caption):
    col_img, col_hdr = st.columns([1, 8])
    with col_img:
        st.markdown(headshot_html(player_id), unsafe_allow_html=True)
    with col_hdr:
        st.subheader(title)
        st.caption(caption)


def traded_banner(team, html_message):
    """A banner in the given team's colours (readable text colour chosen automatically)."""
    bg, border = get_team_color(team, "primary"), get_team_color(team, "secondary")
    r, g, b = (int(bg[i:i + 2], 16) for i in (1, 3, 5))
    txt = "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.5 else "#ffffff"
    st.markdown(
        f"""<div style="background:{bg};border-left:5px solid {border};
            padding:12px 16px;border-radius:6px;color:{txt};
            font-size:15px;margin-bottom:8px;">{html_message}</div>""",
        unsafe_allow_html=True,
    )


def csv_download(label, df, file_name, **to_csv_kwargs):
    st.download_button(label, data=df.to_csv(**to_csv_kwargs), file_name=file_name, mime="text/csv")


def slug(name):
    return name.replace(" ", "_")


def rankings_table(results, actual_team, columns, context_window=5):
    """
    Rank + chosen columns of a team-ranking frame as a scrollable HTML table
    that opens centred on the actual team's (highlighted) row.
    `columns` maps result column → display name. Returns the display frame
    (with an _is_actual column) for CSV export.
    """
    display = results[list(columns) + ["is_actual"]].copy()
    display.columns = list(columns.values()) + ["_is_actual"]
    for col in list(columns.values())[1:]:
        display[col] = display[col].round(3)
    display.insert(0, "Rank", range(1, len(display) + 1))

    actual_idx = display.index[display["_is_actual"]].tolist()
    rank_val   = int(display.loc[actual_idx[0], "Rank"]) if actual_idx else "?"
    _scrollable_table(display.drop(columns=["_is_actual"]).reset_index(drop=True),
                      display["_is_actual"].reset_index(drop=True),
                      actual_team, rank_val, len(display), context_window, get_team_color(actual_team))
    return display


def _scrollable_table(render_df, is_actual, actual_team, rank_val, total, context_window, team_color):
    table_id = "tbl_" + uuid.uuid4().hex[:8]
    th_style = ("padding:6px 12px; text-align:left; background:#1a1a2e; color:#aaa; font-size:13px; "
                "border-bottom:1px solid #333; position:sticky; top:0; z-index:1;")
    headers = "".join(f'<th style="{th_style}">{c}</th>' for c in render_df.columns)

    rows_html = ""
    for i, (_, row) in enumerate(render_df.iterrows()):
        if is_actual.iloc[i]:
            # Team colour at ~35% opacity reads clearly on the dark background
            row_style = f"background:{team_color}59; font-weight:bold; color:#fff; font-size:13.5px;"
            row_id    = f'id="actual_row_{table_id}"'
            td_style  = "padding:6px 14px; font-size:13.5px; border-bottom:1px solid #1f2937;"
        else:
            row_style = "background:#0e1117; color:#ccc;" if i % 2 == 0 else "background:#111827; color:#ccc;"
            row_id    = ""
            td_style  = "padding:5px 12px; font-size:13px; border-bottom:1px solid #1f2937;"
        cells = ""
        for col, v in zip(render_df.columns, row):
            if col == "Team":
                v = (f'<img src="{NHL_LOGO_URL.format(team=v)}" height="20" '
                     f'style="vertical-align:middle;margin-right:6px" '
                     f'onerror="this.style.display=\'none\'"> {v}')
            cells += f'<td style="{td_style}">{v}</td>'
        rows_html += f'<tr style="{row_style}" {row_id}>{cells}</tr>\n'

    row_h = header_h = 38
    height = header_h + (context_window * 2 + 1) * row_h
    html = f"""
<div id="wrap_{table_id}" style="overflow-y:auto; height:{height}px; border:1px solid #2d3748; border-radius:4px;">
  <table id="{table_id}" style="width:100%; border-collapse:collapse; table-layout:auto;">
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
<script>
  (function() {{
    var wrap = document.getElementById("wrap_{table_id}");
    var row  = document.getElementById("actual_row_{table_id}");
    if (wrap && row) {{
      // Scroll the container only — never the page
      wrap.scrollTop = row.offsetTop - (wrap.clientHeight / 2) + (row.offsetHeight / 2);
    }}
  }})();
</script>
"""
    st.caption(f"**{actual_team}** ranks **{rank_val} of {total}** — highlighted row is centred, scroll to see all teams.")
    components.html(html, height=height + 4, scrolling=False)
