# app_direction_colors.py
import re
import pandas as pd
import numpy as np
from textwrap import wrap
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px

# =========================
# ---- Column aliases -----
# =========================
COL_L1 = "INTERVENTION_Level 1_One Health Capability"
COL_L2 = "INTERVENTION_Level 2"

COL_INT_OUT = "INTERMEDIATE OUTCOME_Classification (dropdown)\n"
COL_FIN_OUT = "OUTCOME_Classification (dropdown)\n\n"

COL_INT_STRENGTH = "_strength"
COL_FIN_STRENGTH = "_strength_final"
COL_INT_SIGN = "_sign"
COL_FIN_SIGN = "_sign_final"

# ---- Impact block ----
COL_IMP_OUT = "IMPACT_Classification (dropdown)\nAMR burden reduced = Impact"
COL_IMP_STRENGTH = "_strength_impact"
COL_IMP_SIGN = "_sign_impact"
COL_IMP_TEXT = "IMPACT TEXT (verbatim)"

COL_TITLE = "Title​ \n(213 articles)"
COL_YEAR = "Publication Year​"
COL_GEOG = "Geography​_Location"
COL_INT_TEXT = "INTERMEDIATE OUTCOME TEXT (verbatim)"
COL_INTERV_TEXT = "INTERVENTION TEXT (verbatim)\n"
COL_OUT_TEXT = "OUTCOME TEXT (verbatim)"

# =========================
# ---- LOAD YOUR DATA -----
# =========================
df = pd.read_parquet("df.parquet")

def _clean(s):
    return s.strip() if isinstance(s, str) else s

for c in [COL_L1, COL_L2, COL_INT_OUT, COL_FIN_OUT, COL_IMP_OUT]:
    if c in df.columns:
        df[c] = df[c].map(_clean)

# =========================
# ---- Long-format table ---
# =========================
keep_base = [
    COL_L1, COL_L2, COL_TITLE, COL_YEAR, COL_GEOG,
    COL_INT_TEXT, COL_INTERV_TEXT, COL_OUT_TEXT, COL_IMP_TEXT
]
cols_keep = [c for c in keep_base if c in df.columns]

# Intermediate
int_cols = cols_keep + [COL_INT_OUT, COL_INT_STRENGTH, COL_INT_SIGN]
int_long = (df[int_cols].dropna(subset=[COL_L1, COL_L2, COL_INT_OUT], how="any")
            .assign(outcome_type="Intermediate Outcomes",
                    outcome_name=lambda x: x[COL_INT_OUT],
                    strength=lambda x: x[COL_INT_STRENGTH],
                    sign=lambda x: x[COL_INT_SIGN]))

# Final
fin_cols = cols_keep + [COL_FIN_OUT, COL_FIN_STRENGTH, COL_FIN_SIGN]
fin_long = (df[fin_cols].dropna(subset=[COL_L1, COL_L2, COL_FIN_OUT], how="any")
            .assign(outcome_type="Final Outcomes",
                    outcome_name=lambda x: x[COL_FIN_OUT],
                    strength=lambda x: x[COL_FIN_STRENGTH],
                    sign=lambda x: x[COL_FIN_SIGN]))

# Impact (si faltaran columnas, crea bloque vacío)
if {COL_IMP_OUT, COL_IMP_STRENGTH, COL_IMP_SIGN}.issubset(df.columns):
    imp_cols = cols_keep + [COL_IMP_OUT, COL_IMP_STRENGTH, COL_IMP_SIGN]
    imp_long = (df[imp_cols].dropna(subset=[COL_L1, COL_L2, COL_IMP_OUT], how="any")
                .assign(outcome_type="Impact",
                        outcome_name=lambda x: x[COL_IMP_OUT],
                        strength=lambda x: x[COL_IMP_STRENGTH],
                        sign=lambda x: x[COL_IMP_SIGN]))
else:
    imp_long = pd.DataFrame(columns=int_long.columns)

long = pd.concat([int_long, fin_long, imp_long], ignore_index=True)

# =========================
# ---- Aggregation --------
# =========================
agg = (long.groupby([COL_L1, COL_L2, "outcome_type", "outcome_name"], dropna=False)
       .agg(n=("outcome_name", "size"),
            mean_strength=("strength", "mean"),
            mean_direction=("sign", "mean"))
       .reset_index())

# =========================
# ---- Utils --------------
# =========================
HDR_PREFIX_Y = "__HDR__"
ROW_PREFIX_Y = "__ROW__"
HDR_PREFIX_X = "__XHDR__"
COL_PREFIX_X = "__XCOL__"

INDENT = "\u2003\u2003"     # 2 EM spaces
BULLET = "•"

def wrap_lines(s, width=40, max_lines=3):
    s = "" if pd.isna(s) else str(s)
    lines = wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines-1] + [" ".join(lines[max_lines-1:])]
    return "\n".join(lines)

def make_row_label(s, width=40, max_lines=3):
    return f"{INDENT}{BULLET} {wrap_lines(s, width, max_lines)}"

def l2_num_prefix(s):
    m = re.match(r"\s*(\d+)[\.\)]", str(s))
    return int(m.group(1)) if m else 9999

# ---- Etiquetas strength & direction (para hover/tabla) ----
def map_strength_label(x):
    if pd.isna(x): return "N/A"
    if x >= 2.6:   return "high"
    elif x >= 2.4: return "moderate"
    elif x >= 1.5: return "low-moderate"
    elif x >= 1.0: return "low"
    else:          return "very low"

def map_direction_label(x):
    if pd.isna(x): return "N/A"
    if x == 1.0:   return "positive"
    elif 0.5 < x < 1: return "mostly positive"
    elif -0.5 < x <= 0.5: return "inconclusive/mixed"
    elif -1 < x <= -0.5: return "mostly negative"
    elif x == -1.0: return "negative"

agg["strength_label"]  = agg["mean_strength"].map(map_strength_label)
agg["direction_label"] = agg["mean_direction"].map(map_direction_label)

# =========================
# ---- Hierarchical Y axis
# =========================
unique_l1 = list(agg[COL_L1].dropna().unique())
unique_l1 = [l for l in unique_l1 if str(l).lower().startswith("surveillance")] + \
            [l for l in unique_l1 if not str(l).lower().startswith("surveillance")]

agg["y_key"] = ROW_PREFIX_Y + agg[COL_L1].astype(str) + " || " + agg[COL_L2].astype(str)

order_y, y_ticktext, y_header_keys = [], [], []
for l1 in unique_l1:
    grp = agg[agg[COL_L1] == l1].copy()
    hdr_key = HDR_PREFIX_Y + str(l1)
    order_y.append(hdr_key)
    y_ticktext.append(" ")
    y_header_keys.append(hdr_key)
    grp = grp.sort_values(by=COL_L2, key=lambda s: s.map(l2_num_prefix))
    for _, r in grp.iterrows():
        order_y.append(r["y_key"])
        y_ticktext.append(make_row_label(r[COL_L2], width=40, max_lines=3))

order_y_rev  = order_y[::-1]
y_ticktext_rev = y_ticktext[::-1]

# =========================
# ---- Hierarchical X axis
# =========================
agg["x_key"] = COL_PREFIX_X + agg["outcome_type"] + " || " + agg["outcome_name"].astype(str)

order_x, x_ticktext, x_header_keys = [], [], []
x_items_by_type = {}

for t in ["Intermediate Outcomes", "Final Outcomes", "Impact"]:
    hdr = HDR_PREFIX_X + t
    order_x.append(hdr)
    x_ticktext.append("")
    x_header_keys.append(hdr)

    subnames = (agg.loc[agg["outcome_type"] == t, "outcome_name"]
                  .astype(str).sort_values().unique().tolist())
    x_items_by_type[t] = [COL_PREFIX_X + t + " || " + name for name in subnames]

    for key, name in zip(x_items_by_type[t], subnames):
        order_x.append(key)
        x_ticktext.append(make_row_label(name, width=24, max_lines=3))

# =========================
# ---- Labels para hover ---
# =========================
agg["Intervention_label"] = agg[COL_L1].astype(str) + " \u2192 " + agg[COL_L2].astype(str)
agg["Outcome_label"] = agg["outcome_name"].astype(str)

# =========================
# ---- Sizing -------------
# =========================
n_rows = max(1, len(order_y))
fig_height = int(min(1400, 120 + 24 * n_rows))
fig_width = 1650
desired_max_px = 48
max_n = max(1, agg["n"].max())
sizeref = 2.0 * max_n / (desired_max_px ** 2)

# =========================
# ---- Colors (direction) -
# =========================
# Orden y colores para la leyenda
DIR_ORDER = ["positive", "mostly positive", "inconclusive/mixed", "mostly negative", "negative"]
DIR_COLORS = {
    "positive": "#1b9e77",
    "mostly positive": "#66c2a4",
    "inconclusive/mixed": "#7f7f7f",
    "mostly negative": "#fc8d62",
    "negative": "#d95f02",
}

# =========================
# ---- Figure -------------
# =========================
fig = px.scatter(
    agg,
    x="x_key",
    y="y_key",
    size="n",                            # tamaño = número de artículos
    color="direction_label",             # color = dirección categórica
    category_orders={"direction_label": DIR_ORDER},
    color_discrete_map=DIR_COLORS,
    size_max=desired_max_px,
    custom_data=[
        "Intervention_label",  # 0
        "Outcome_label",       # 1
        "n",                   # 2
        "strength_label",      # 3
        "direction_label"      # 4
    ]
)
fig.update_traces(
    marker=dict(line=dict(width=0), sizemode="area", sizeref=sizeref, sizemin=4),
    hovertemplate=(
        "<b>Intervention</b>: %{customdata[0]}<br>"
        "<b>Result</b>: %{customdata[1]}<br>"
        "<b>Number of articles</b>: %{customdata[2]}<br>"
        "<b>Strength of evidence</b>: %{customdata[3]}<br>"
        "<b>Direction of evidence</b>: %{customdata[4]}"
        "<extra></extra>"
    )
)

fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin=dict(l=520, r=60, t=90, b=330),
    xaxis=dict(
        title="",
        categoryorder="array", categoryarray=order_x,
        tickmode="array", tickvals=order_x, ticktext=x_ticktext,
        tickangle=35, tickfont=dict(size=11),
    ),
    yaxis=dict(
        title="Intervention themes",
        categoryorder="array", categoryarray=order_y_rev,
        tickmode="array", tickvals=order_y_rev, ticktext=y_ticktext_rev,
        automargin=True, tickfont=dict(size=11),
    ),
    legend_title_text="Direction of evidence",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

# ---- Row titles (Level 1)
for hdr_key in y_header_keys:
    l1_text = hdr_key.replace(HDR_PREFIX_Y, "")
    fig.add_annotation(
        xref="paper", yref="y",
        x=-0.0, y=hdr_key,
        text=f"<b>{wrap_lines(l1_text, 38, 3)}</b>",
        showarrow=False, xanchor="right", yanchor="middle",
        align="right", font=dict(size=13), bgcolor="rgba(0,0,0,0)"
    )

# ---- Column block titles (IO / FO / Impact)
N = len(order_x) - 1
for t in ["Intermediate Outcomes", "Final Outcomes", "Impact"]:
    items = x_items_by_type.get(t, [])
    if not items:
        continue
    first_idx = order_x.index(items[0])
    last_idx  = order_x.index(items[-1])
    center_frac = (first_idx + last_idx) / 2 / N
    fig.add_annotation(
        xref="x domain", yref="paper",
        x=center_frac, y=-0.25,
        text=f"<b>{t}</b>",
        textangle=0,
        showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=14)
    )

# =========================
# ---- Dash App ----------
# =========================
app = Dash(__name__)
app.title = "Scaling One Health Capabilities — Direction colored"

app.layout = html.Div([
    html.H2("Scaling One Health Capabilities"),
    html.P("The Strength of Evidence is defined on a scale from 0.5 (very low) to 3 (high). The Direction of Effect is classified as Positive (+1), Inconclusive/Mixed (0), or Negative (–1). Average values can be viewed by clicking on each bubble. Bubble size shows the number of articles. Color encodes the direction of evidence."),
    dcc.Graph(id="bubble-graph", figure=fig, clear_on_unhover=True,
              style={"width": "100%"}),

    html.Hr(),
    html.H4(id="detail-title", children="Click a bubble to see the article list."),
    dash_table.DataTable(
        id="detail-table",
        columns=[
            {"name": "Title", "id": COL_TITLE},
            {"name": "Year", "id": COL_YEAR},
            {"name": "Location", "id": COL_GEOG},
            {"name": "Intervention", "id": COL_INTERV_TEXT},
            {"name": "Int. outcome", "id": COL_INT_TEXT},
            {"name": "Int. outcome - Strength", "id": COL_INT_STRENGTH},
            {"name": "Int. outcome - Direction", "id": COL_INT_SIGN},
            {"name": "Final outcome", "id": COL_OUT_TEXT},
            {"name": "Final outcome - Strength", "id": COL_FIN_STRENGTH},
            {"name": "Final outcome - Direction", "id": COL_FIN_SIGN},
            {"name": "Impact", "id": COL_IMP_TEXT},
            {"name": "Impact - Strength", "id": COL_IMP_STRENGTH},
            {"name": "Impact - Direction", "id": COL_IMP_SIGN},
        ],
        data=[],
        sort_action="native", filter_action="native", page_size=10,
        style_cell={"whiteSpace": "pre-line", "textAlign": "left",
                    "minWidth": "160px", "maxWidth": "420px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_table={"overflowX": "auto"},
    )
], style={"maxWidth": "1800px", "margin": "0 auto", "padding": "12px"})

# =========================
# ---- Callbacks ----------
# =========================
@app.callback(
    Output("detail-table", "data"),
    Output("detail-title", "children"),
    Input("bubble-graph", "clickData"),
)
def show_details(clickData):
    if not clickData:
        return [], "Click a bubble to see the article list."

    x_key = clickData["points"][0]["x"]
    y_key = clickData["points"][0]["y"]

    if not (str(x_key).startswith(COL_PREFIX_X) and str(y_key).startswith(ROW_PREFIX_Y)):
        return [], "Select a bubble (row Level 2 × outcome) to see details."

    # Parse x: "__XCOL__<type> || <outcome>"
    _, rest = x_key.split(COL_PREFIX_X, 1)
    outcome_type, outcome_name = rest.split(" || ", 1)

    # Parse y: "__ROW__<L1> || <L2>"
    _, rest_y = y_key.split(ROW_PREFIX_Y, 1)
    l1, l2 = rest_y.split(" || ", 1)

    mask = ((long[COL_L1] == l1) &
            (long[COL_L2] == l2) &
            (long["outcome_type"] == outcome_type) &
            (long["outcome_name"] == outcome_name))
    sub = long.loc[mask].copy()

    cols_show = [
        COL_TITLE, COL_YEAR, COL_GEOG, COL_INTERV_TEXT,
        COL_INT_TEXT, COL_OUT_TEXT, COL_IMP_TEXT,
        COL_INT_STRENGTH, COL_FIN_STRENGTH, COL_IMP_STRENGTH,
        COL_INT_SIGN, COL_FIN_SIGN, COL_IMP_SIGN
    ]
    for c in cols_show:
        if c not in sub.columns:
            sub[c] = np.nan

    title = f'{outcome_type} — "{outcome_name}"  |  {l1} → {l2}  |  {len(sub)} articles'
    return sub[cols_show].to_dict("records"), title

# =========================
# ---- Run server ---------
# =========================
if __name__ == "__main__":
    app.run(debug=True)



