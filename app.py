"""
app.py
======
NHL Player Predictor — Streamlit UI entry point.
All model logic lives in model_utils.py.

Run with:  python -m streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import textwrap
from streamlit_option_menu import option_menu
from model_utils import *

# ── Module-level callbacks — must be defined before any widgets ─────────────
def _on_insertion_team_change():
    st.session_state["active_team"]    = st.session_state.get("insertion_team")
    st.session_state["_team_override"] = True

def _on_pair_team_change():
    st.session_state["active_team"]    = st.session_state.get("pair_team_sel")
    st.session_state["_team_override"] = True

def _on_contract_team_change():
    st.session_state["active_team"]    = st.session_state.get("contract_team")
    st.session_state["_team_override"] = True

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="NHL Player Predictor", page_icon="🏒", layout="wide", initial_sidebar_state="collapsed")
# Apply theme on every run.
# player_base_team  = the player's actual team  → always used on non-override tabs
# active_team       = insertion / pairing / contract team → used only on those tabs
# The JS observer in _TAB_THEME_JS (injected below) watches aria-selected on tab
# buttons and swaps between --team-bg-base and --team-bg-override in real time.
st.markdown("""
    <style>
        /* Force the app to scroll internally so position:fixed works in HF Spaces */
        html, body {
            height: 100vh !important;
            overflow: hidden !important;
        }
        .main {
            height: 100vh !important;
            overflow-y: auto !important;
        }
        .block-container {
            max-width: 860px;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 3rem;
        }
        /* Pin hamburger button — now truly fixed since page scrolls internally */
        [data-testid="collapsedControl"] {
            position: fixed !important;
            top: 10px !important;
            left: 10px !important;
            z-index: 999999 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(30,30,30,0.85) !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
        }
        [data-testid="collapsedControl"] svg {
            display: block !important;
            color: white !important;
            fill: white !important;
        }
        section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] svg { display:none !important; }
    </style>
""", unsafe_allow_html=True)
apply_team_theme(
    player_team   = st.session_state.get("player_base_team"),
    override_team = st.session_state.get("active_team") if st.session_state.get("_team_override") else None,
)
st.title("NHL Player Predictor")

if "fit_models" not in st.session_state:
    if os.path.exists(CACHE_FILE):
        with st.spinner("Loading saved models from disk..."):
            cached = joblib.load(CACHE_FILE)
            (
                st.session_state["df"],
                st.session_state["team_ctx"],
                st.session_state["has_age"],
                st.session_state["player_profiles"],
                st.session_state["fit_models"],
                st.session_state["fit_metrics"],
                st.session_state["fit_feature_names"],
                st.session_state["next_models"],
                st.session_state["next_metrics"],
                st.session_state["next_feature_names"],
            ) = cached
    else:
        st.info("Training models for the first time — this takes 5–8 minutes. Won't happen again until you retrain.")
        results = load_and_train_with_progress(DATA_FILE, AGES_FILE)
        joblib.dump(results, CACHE_FILE)
        (
            st.session_state["df"],
            st.session_state["team_ctx"],
            st.session_state["has_age"],
            st.session_state["player_profiles"],
            st.session_state["fit_models"],
            st.session_state["fit_metrics"],
            st.session_state["fit_feature_names"],
            st.session_state["next_models"],
            st.session_state["next_metrics"],
            st.session_state["next_feature_names"],
        ) = results
        st.rerun()

df                 = st.session_state["df"]
team_ctx           = st.session_state["team_ctx"]
has_age            = st.session_state["has_age"]
player_profiles    = st.session_state["player_profiles"]
fit_models         = st.session_state["fit_models"]
fit_metrics        = st.session_state["fit_metrics"]
fit_feature_names  = st.session_state["fit_feature_names"]
next_models        = st.session_state["next_models"]
next_metrics       = st.session_state["next_metrics"]
next_feature_names = st.session_state["next_feature_names"]

# ── Load / train defensive model ───────────────────────────────────────────────
if "def_fit_models" not in st.session_state:
    if os.path.exists(DEF_CACHE_FILE):
        with st.spinner("Loading saved defensive models..."):
            def_cached = joblib.load(DEF_CACHE_FILE)
            (
                st.session_state["def_df"],
                st.session_state["def_team_ctx"],
                st.session_state["def_has_age"],
                st.session_state["def_player_profiles"],
                st.session_state["def_fit_models"],
                st.session_state["def_fit_metrics"],
                st.session_state["def_fit_feature_names"],
                st.session_state["def_next_models"],
                st.session_state["def_next_metrics"],
                st.session_state["def_next_feature_names"],
            ) = def_cached

    elif os.path.exists(DEF_FILE):
        st.info("Training defensive models for the first time — takes 3-5 minutes.")
        def_results = def_load_and_train(DEF_FILE, AGES_FILE)
        joblib.dump(def_results, DEF_CACHE_FILE)
        (
            st.session_state["def_df"],
            st.session_state["def_team_ctx"],
            st.session_state["def_has_age"],
            st.session_state["def_player_profiles"],
            st.session_state["def_fit_models"],
            st.session_state["def_fit_metrics"],
            st.session_state["def_fit_feature_names"],
            st.session_state["def_next_models"],
            st.session_state["def_next_metrics"],
            st.session_state["def_next_feature_names"],
        ) = def_results
        st.rerun()
    else:
        st.session_state["def_fit_models"] = None

def_models_loaded = st.session_state.get("def_fit_models") is not None

if def_models_loaded:
    def_df                = st.session_state["def_df"]
    def_team_ctx          = st.session_state["def_team_ctx"]
    def_has_age           = st.session_state["def_has_age"]
    def_player_profiles   = st.session_state["def_player_profiles"]
    def_fit_models        = st.session_state["def_fit_models"]
    def_fit_metrics       = st.session_state["def_fit_metrics"]
    def_fit_feature_names = st.session_state["def_fit_feature_names"]
    def_next_models       = st.session_state["def_next_models"]
    def_next_metrics      = st.session_state["def_next_metrics"]
    def_next_feature_names= st.session_state["def_next_feature_names"]

if has_age:
    st.markdown(
        '<div style="display:block;background:#1a4a2e;border:1px solid #2ecc71;'
        'border-radius:20px;padding:6px 18px;font-size:13px;">'
        '<span style="color:#2ecc71 !important;font-weight:700;">● Age data loaded</span>'
        '<span style="color:#cccccc !important;"> — next-season forecasting active</span>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="display:block;background:#4a3000;border:1px solid #f39c12;'
        'border-radius:20px;padding:6px 18px;font-size:13px;">'
        '<span style="color:#f39c12 !important;font-weight:700;">⚠ Age data not found</span>'
        '<span style="color:#cccccc !important;"> — next-season model running without age features</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Hamburger CSS (hides arrows; JS in _TOUR_HTML does the patching) ─────────
st.markdown("""
    <style>
        [data-testid="collapsedControl"] svg { display:none !important; }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    active_tab = option_menu(
        menu_title=None,
        options=[
            "NHL Predictor",
            "Introduction",
            "Literature Review",
            "Methodology",
            "Analysis & Findings",
            "Conclusion",
            "Works Cited",
        ],
        icons=[""] * 7,
        default_index=0,
        styles={"icon": {"display": "none"}},
    )

# ── NHL Predictor (all app content) ──────────────────────────────────────────
if active_tab == "NHL Predictor":
    # ── Pre-build filtered player lists ──────────────────────────────────────────
    # Filter to players active within the last 3 seasons
    _current_season = int(df["season"].max())
    _cutoff_season  = _current_season - 2  # must have played in last 3 seasons

    _fwd_latest = (
        df.groupby("player_name")["season"].max()
    )
    _active_fwd = set(_fwd_latest[_fwd_latest >= _cutoff_season].index)

    if def_models_loaded:
        _def_latest = def_df.groupby("player_name")["season"].max()
        _cutoff_def = int(def_df["season"].max()) - 2
        _active_def = set(_def_latest[_def_latest >= _cutoff_def].index)
    else:
        _active_def = set()

    # Name → position maps for dropdown labels
    _fwd_pos_map = (
        df[df["player_name"].isin(_active_fwd)]
          .sort_values("season", ascending=False)
          .drop_duplicates("player_name")
          .set_index("player_name")["position"]
          .to_dict()
    )
    _def_pos_map = {name: "D" for name in _active_def} if def_models_loaded else {}
    _all_pos_map = {**_fwd_pos_map, **_def_pos_map}

    _fwd_names = sorted(_fwd_pos_map.keys())
    _def_names = sorted(_def_pos_map.keys()) if def_models_loaded else []
    _all_names = sorted(set(_fwd_names) | set(_def_names))

    def _fmt_fwd(name):
        if not name: return ""
        return f"{name}  ({_fwd_pos_map.get(name, 'F')})"

    def _fmt_def(name):
        if not name: return ""
        return f"{name}  (D)"

    def _fmt_all(name):
        if not name: return ""
        return f"{name}  ({_all_pos_map.get(name, '?')})"

    # Shared pred — populated by whichever tab last ran a search.
    # Contract tab uses this since it handles both positions.
    if "shared_pred" not in st.session_state:
        st.session_state["shared_pred"] = None

    # ── Predictor sub-tabs ────────────────────────────────────────────────────────
    tab_off, tab_def, tab_contract, tab_model, tab_val = st.tabs([
        "Offensive",
        "Defensive",
        "Contract Evaluator",
        "Models",
        "Validation",
    ])

    # ── Interactive spotlight tour ────────────────────────────────────────────────
    # iframe height=0 — invisible. Script immediately injects the fixed button
    # and all overlay divs into window.parent.document.body so nothing appears
    # in the page scroll flow.
    _TOUR_HTML = """
<!DOCTYPE html>
<html>
<head><style>body{margin:0;background:transparent;}</style></head>
<body>
<script>
(function(){
var P = window.parent;
var D = P.document;
var curr = 0;
var waitPoll = null;

if (D.getElementById('tc-btn')) return; // already injected on re-render

// ── Hamburger / X patcher ────────────────────────────────────────────────────
function patchSidebarBtns() {
  // Collapsed sidebar toggle → ☰
  var coll = D.querySelector('[data-testid="collapsedControl"]');
  if (coll && !coll.dataset.hPatched) {
    coll.dataset.hPatched = '1';
    coll.style.fontSize = '22px';
    coll.style.display = 'flex';
    coll.style.alignItems = 'center';
    coll.style.justifyContent = 'center';
    var s = D.createElement('span');
    s.textContent = '\u2630';
    s.style.cssText = 'font-size:22px;line-height:1;pointer-events:none;';
    coll.appendChild(s);
  }
  // Open sidebar close button → ✕
  var sb = D.querySelector('section[data-testid="stSidebar"]');
  if (sb) {
    var cb = sb.querySelector('[data-testid="baseButton-headerNoPadding"]');
    if (cb && !cb.dataset.hPatched) {
      cb.dataset.hPatched = '1';
      cb.style.fontSize = '18px';
      cb.style.display = 'flex';
      cb.style.alignItems = 'center';
      cb.style.justifyContent = 'center';
      var x = D.createElement('span');
      x.textContent = '\u2715';
      x.style.cssText = 'font-size:18px;line-height:1;pointer-events:none;';
      cb.appendChild(x);
    }
  }
}
patchSidebarBtns();
var hObs = new P.MutationObserver(patchSidebarBtns);
hObs.observe(D.body, { childList: true, subtree: true });

var style = D.createElement('style');
style.textContent =
  '#tc-btn{position:fixed;bottom:70px;right:16px;z-index:99997;' +
    'background:linear-gradient(135deg,#3b82f6,#1d4ed8);' +
    'color:#fff;border:none;padding:11px 22px;border-radius:50px;' +
    'cursor:pointer;font-size:14px;font-weight:700;letter-spacing:.3px;' +
    'box-shadow:0 4px 18px rgba(59,130,246,.5);' +
    'transition:transform .15s,box-shadow .15s;}' +
  '#tc-btn:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(59,130,246,.6);}' +
  '#tc-backdrop{position:fixed;inset:0;z-index:99998;display:none;pointer-events:none;}' +
  '#tc-spot{position:fixed;z-index:99999;pointer-events:none;display:none;' +
    'border-radius:8px;outline:3px solid #60a5fa;outline-offset:4px;' +
    'box-shadow:0 0 0 9999px rgba(0,0,0,.75);' +
    'transition:top .3s,left .3s,width .3s,height .3s;}' +
  '#tc-card{position:fixed;z-index:100000;display:none;width:340px;' +
    'background:#0f172a;border:1px solid #1e40af;border-radius:14px;' +
    'padding:20px 22px;color:#f1f5f9;' +
    'box-shadow:0 16px 48px rgba(0,0,0,.7);' +
    'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;' +
    'transition:top .3s,left .3s;}' +
  '#tc-card .lbl{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px;}' +
  '#tc-card h3{margin:0 0 8px;font-size:15px;color:#93c5fd;font-weight:700;}' +
  '#tc-card p{margin:0 0 16px;font-size:13.5px;line-height:1.6;color:#cbd5e1;}' +
  '#tc-card .prog{display:flex;gap:4px;margin-bottom:14px;}' +
  '.tc-dot{height:4px;border-radius:2px;flex:1;background:#1e293b;transition:background .25s;}' +
  '.tc-dot.on{background:#3b82f6;}.tc-dot.dn{background:#1d4ed8;}' +
  '#tc-card .btns{display:flex;gap:8px;justify-content:flex-end;align-items:center;}' +
  '.tb{padding:7px 15px;border-radius:7px;border:none;cursor:pointer;font-size:13px;font-weight:600;}' +
  '.tx{background:transparent;color:#64748b;border:1px solid #334155 !important;}' +
  '.tp{background:#1e293b;color:#cbd5e1;}.tp:hover{background:#334155;}' +
  '.tn{background:#3b82f6;color:#fff;}.tn:hover{background:#2563eb;}';
D.head.appendChild(style);

var btn = D.createElement('button');
btn.id = 'tc-btn';
btn.innerHTML = '&#x1F5FA;&#xFE0F;&nbsp; Tour the App';
btn.onclick = initAndStart;
D.body.appendChild(btn);

var bd = D.createElement('div'); bd.id = 'tc-backdrop'; D.body.appendChild(bd);
var sp = D.createElement('div'); sp.id = 'tc-spot';     D.body.appendChild(sp);
var cd = D.createElement('div'); cd.id = 'tc-card';     D.body.appendChild(cd);
// backdrop is pointer-events:none so user can interact with the page freely
// Press Escape or use the Exit button to end the tour

// ── Per-tab step sets ────────────────────────────────────────────────────────
var ALL_TOURS = {

  'Offensive': [
    { find: function(){ return subTabByText('Offensive'); },
      title: 'Offensive Tab &#x2694;&#xFE0F;',
      text: "This is your hub for evaluating NHL forwards. Everything here is powered by a LightGBM model trained on MoneyPuck shot-tracking data.",
      pos: 'below' },
    { find: function(){
        return findSelectByLabel('forward');
      },
      title: 'Player Search &#x1F50D;',
      text: "Type any part of a forward's name &#x2014; partial names work fine. The dropdown auto-completes from the full player database.",
      pos: 'below' },
    { find: function(){ return subTabByText('Team Fit'); },
      title: 'Team Fit &#x1F4CA;',
      text: "Ranks all 32 NHL teams by how well they'd fit this forward right now &#x2014; showing predicted Points/GP, Goals/GP, and Game Score. Gold = best fit.",
      pos: 'below' },
    { find: function(){ return subTabByText('Next Season'); },
      title: 'Next Season &#x1F4C8;',
      text: "Projects the forward's production into next season using empirical age curves and trajectory signals (rising vs. declining).",
      pos: 'below' },
    { find: function(){ return subTabByText('Roster Insertion'); },
      title: 'Roster Insertion &#x1F4CB;',
      text: "Pick any NHL team to see exactly where this forward slots into the lineup &#x2014; 1st through 4th line &#x2014; and which players get pushed down. A CSV download is available at the bottom.",
      pos: 'below' }
  ],

  'Defensive': [
    { find: function(){ return subTabByText('Defensive'); },
      title: 'Defensive Tab &#x1F6E1;&#xFE0F;',
      text: "This is your hub for evaluating NHL defensemen. The model automatically classifies each D-man as Defensive D, Offensive D, or Two-Way D.",
      pos: 'below' },
    { find: function(){
        return findSelectByLabel('defenseman');
      },
      title: 'Defenseman Search &#x1F50D;',
      text: "Type any part of a defenseman's name here. Once selected, the player's archetype and stat grades appear automatically.",
      pos: 'below' },
    { find: function(){ return firstVisibleTab('Team Fit'); },
      title: 'Team Fit &#x1F4CA;',
      text: "Shows defensive grade (hits, takeaways, xGA, PIM) and offensive grade across all 32 teams. Use this to find the best organizational fit.",
      pos: 'below' },
    { find: function(){ return firstVisibleTab('Next Season'); },
      title: 'Next Season &#x1F4C8;',
      text: "Multi-year defensive forecast using age curves specific to defensemen, who peak and decline on a different timeline than forwards.",
      pos: 'below' },
    { find: function(){ return firstVisibleTab('Pairing'); },
      title: 'Pairing Tool &#x1F91D;',
      text: "Pick any NHL team to see the full defensive depth chart after inserting this player. Pairs are anchored by real shift data. A &#x1F91D; icon means opposite-hand pairing &#x2014; generally preferred by coaches. Download the full chart as a CSV.",
      pos: 'below' }
  ],

  'Contract Evaluator': [
    { find: function(){ return subTabByText('Contract Evaluator'); },
      title: 'Contract Evaluator &#x1F4C4;',
      text: "Projects a player's production across multiple seasons using empirical NHL age curves. Works for both forwards and defensemen.",
      pos: 'below' },
    { find: function(){
        // Find the visible "Search for a player" label's parent container
        var label = Array.from(D.querySelectorAll('label')).find(function(l){
          return l.textContent.trim().toLowerCase().includes('search for a player');
        });
        if (label && label.parentElement) return label.parentElement;
        // Fallback: first selectbox container
        return D.querySelector('[data-testid="stSelectbox"]')
            || D.querySelector('.stSelectbox');
      },
      title: 'Search for a Player &#x1F50D;',
      text: "Type any forward or defenseman's name and select them from the dropdown.",
      pos: 'below',
      waitFor: function(){
        // Only fire once elements that ONLY appear after player selection exist.
        // A slider (contract length) is the most reliable signal.
        if (D.querySelector('input[type="range"]')) return true;
        if (D.querySelector('[data-testid="stSlider"]')) return true;
        if (D.querySelector('.stSlider')) return true;
        return false;
      },
      waitMsg: "&#x23F3; Select a player above to continue the tour…"
    },
    { find: function(){
        var H = P.innerHeight;
        var candidates = Array.from(D.querySelectorAll(
          '[data-baseweb="select"], [data-testid="stSelectbox"], .stSelectbox'
        ));
        // First find the player search box (the one containing the selected player name)
        var playerBox = candidates.find(function(el) {
          var r = el.getBoundingClientRect();
          return r.width > 30 && r.height > 10 && r.top >= 0 && r.bottom <= H;
        });
        if (!playerBox) return null;
        var playerBottom = playerBox.getBoundingClientRect().bottom;
        // Return the first visible select that starts BELOW the player search box
        return candidates.find(function(el) {
          if (el === playerBox) return false;
          var r = el.getBoundingClientRect();
          return r.width > 30 && r.height > 10 && r.top > playerBottom && r.bottom <= H + 200;
        }) || null;
      },
      title: 'Team Selection &#x1F3D2;',
      text: "Pick the team you want to evaluate the contract for. The model adjusts its projection based on that team's playing style and deployment tendencies.",
      pos: 'below' },
    { find: function(){
        return D.querySelector('input[type="range"]')
            || D.querySelector('.stSlider')
            || D.querySelector('[data-testid="stSlider"]')
            || D.querySelector('.stNumberInput');
      },
      title: 'Contract Length &#x1F4C5;',
      text: "Set how many years to project &#x2014; from 1 to 8 seasons. The model applies NHL age curves to each year so you can see when production is expected to peak or decline.",
      pos: 'below' },
    { find: function(){
        return D.querySelector('.js-plotly-plot')
            || D.querySelector('[data-testid="stVegaLiteChart"]')
            || D.querySelector('.stDataFrame')
            || D.querySelector('[data-testid="stArrowDataFrame"]')
            || D.querySelector('[data-testid="stTable"]');
      },
      title: 'Production Projection &#x1F4C8;',
      text: "The projection shows predicted output for each season of the deal. Confidence bands widen in later years &#x2014; treat years 4+ as a range, not a precise number.",
      pos: 'below' }
  ],

  'Models': [
    { find: function(){ return subTabByText('Models'); },
      title: 'Models Tab &#x1F52C;',
      text: "See exactly how well the prediction models perform before trusting any output.",
      pos: 'below' },
    { find: function(){ return D.querySelector('.stMetric') || D.querySelector('[data-testid="metric-container"]'); },
      title: 'Accuracy Metrics &#x1F4CF;',
      text: "MAE and RMSE are shown for Points/GP, Goals/GP, and Game Score/GP &#x2014; measured via 3-fold cross-validation on held-out seasons. Lower is better.",
      pos: 'below' },
    { find: function(){ return D.querySelector('.js-plotly-plot') || D.querySelector('[data-testid="stVegaLiteChart"]'); },
      title: 'Feature Importance &#x1F4CA;',
      text: "Shows which variables drive predictions most &#x2014; e.g. high-danger shot share, age &#xD7; finishing skill. Use this to understand what the model is actually learning.",
      pos: 'below' }
  ],

  'Validation': [
    { find: function(){ return subTabByText('Validation'); },
      title: 'Validation Tab &#x2705;',
      text: "Back-test the model against historical seasons where the true outcome is already known.",
      pos: 'below' },
    { find: function(){ return D.querySelector('[data-baseweb="select"]') || D.querySelector('.stSelectbox'); },
      title: 'Select Season & Player &#x1F50D;',
      text: "Pick a past season and a player. The app shows what the model would have predicted at the time alongside the player's actual stat line.",
      pos: 'below' }
  ]
};

// Fallback full tour if no tab is matched
var FALLBACK_STEPS = [
  { find: function(){ return tabByText('NHL Predictor'); },
    title: 'Welcome &#x1F3D2;',
    text: "Everything lives inside the NHL Predictor tab. Navigate to any sub-tab and click Tour the App for a focused walkthrough of that section.",
    pos: 'below' },
  { find: function(){ return subTabByText('Offensive'); },
    title: 'Offensive Tab &#x2694;&#xFE0F;',
    text: "Start here to evaluate NHL forwards across all 32 teams.",
    pos: 'below' },
  { find: function(){ return subTabByText('Defensive'); },
    title: 'Defensive Tab &#x1F6E1;&#xFE0F;',
    text: "Switch here for defensemen, including the Pairing tool.",
    pos: 'below' },
  { find: function(){ return subTabByText('Contract Evaluator'); },
    title: 'Contract Evaluator &#x1F4C4;',
    text: "Project any player over 1&#x2013;8 seasons to evaluate a contract offer.",
    pos: 'below' },
  { find: function(){ return subTabByText('Models'); },
    title: 'Models &#x1F52C;',
    text: "Review model accuracy metrics and feature importance.",
    pos: 'below' },
  { find: function(){ return subTabByText('Validation'); },
    title: 'Validation &#x2705;',
    text: "Back-test predictions against seasons with known outcomes.",
    pos: 'below' }
];

var STEPS = FALLBACK_STEPS; // active step set, set on tour start

function tabByText(t) {
  return Array.from(D.querySelectorAll('button[role="tab"]'))
    .find(function(b){ return b.textContent.includes(t); });
}
function subTabByText(t) {
  var all = Array.from(D.querySelectorAll('button[role="tab"]'));
  return all.find(function(b){ return b.textContent.trim() === t; })
      || all.find(function(b){ return b.textContent.includes(t); });
}

// Walk up to body checking display:none, visibility:hidden and aria-hidden.
// getBoundingClientRect alone is unreliable: Streamlit can use visibility:hidden
// or aria-hidden on inactive tab panels, leaving the rect non-zero.
function isTrulyVisible(el) {
  var cur = el;
  while (cur && cur !== D.body) {
    if (cur.getAttribute('aria-hidden') === 'true') return false;
    if (cur.hasAttribute('hidden')) return false;
    var cs = P.getComputedStyle(cur);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    cur = cur.parentElement;
  }
  var r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}

// Find the Streamlit selectbox whose label contains needleText AND is visible.
// Targets [data-testid="stSelectbox"] (the full widget) rather than
// [data-baseweb="select"] (just the inner trigger) for reliable dimensions.
function findSelectByLabel(needleText) {
  var lc = needleText.toLowerCase();
  return Array.from(D.querySelectorAll('[data-testid="stSelectbox"]')).find(function(box) {
    var lbl = box.querySelector('label');
    return lbl && lbl.textContent.toLowerCase().includes(lc) && isTrulyVisible(box);
  }) || firstVisible('[data-testid="stSelectbox"]');
}

// First element matching CSS selector that passes the full visibility check.
function firstVisible(selector) {
  return Array.from(D.querySelectorAll(selector)).find(isTrulyVisible) || null;
}

// First tab button whose text matches AND whose ancestors are all visible.
function firstVisibleTab(text) {
  return Array.from(D.querySelectorAll('button[role="tab"]')).find(function(b) {
    return b.textContent.trim().includes(text) && isTrulyVisible(b);
  }) || null;
}

// Allow Escape key to exit tour
D.addEventListener('keydown', function(e) {
  if (e.key === 'Escape' && D.getElementById('tc-card') &&
      D.getElementById('tc-card').style.display !== 'none') {
    exitTour();
  }
});

function initAndStart() {
  // Detect which sub-tab is currently active
  var activeTab = null;
  var tabCandidates = ['Offensive', 'Defensive', 'Contract Evaluator', 'Models', 'Validation'];
  for (var t = 0; t < tabCandidates.length; t++) {
    var tabEl = subTabByText(tabCandidates[t]);
    if (tabEl && (tabEl.getAttribute('aria-selected') === 'true' || tabEl.classList.contains('st-d8'))) {
      activeTab = tabCandidates[t];
      break;
    }
  }
  // Also check by which tab button has the active underline style
  if (!activeTab) {
    var allTabs = Array.from(D.querySelectorAll('button[role="tab"]'));
    for (var j = 0; j < allTabs.length; j++) {
      var b = allTabs[j];
      if (b.getAttribute('aria-selected') === 'true') {
        var txt = b.textContent.trim();
        if (ALL_TOURS[txt]) { activeTab = txt; break; }
      }
    }
  }
  STEPS = (activeTab && ALL_TOURS[activeTab]) ? ALL_TOURS[activeTab] : FALLBACK_STEPS;

  // Update button label to show context
  var label = activeTab ? ('Tour: ' + activeTab) : 'Tour the App';
  D.getElementById('tc-btn').innerHTML = '&#x1F5FA;&#xFE0F;&nbsp; ' + label;

  curr = 0;
  D.getElementById('tc-backdrop').style.display = 'block';
  showStep(0);
}

// Update button label whenever a sub-tab is clicked
setTimeout(function attachTabListeners() {
  var tabBtns = Array.from(D.querySelectorAll('button[role="tab"]'));
  tabBtns.forEach(function(b) {
    b.addEventListener('click', function() {
      var txt = b.textContent.trim();
      var lbl = ALL_TOURS[txt] ? ('Tour: ' + txt) : 'Tour the App';
      var btn = D.getElementById('tc-btn');
      if (btn) btn.innerHTML = '&#x1F5FA;&#xFE0F;&nbsp; ' + lbl;
    });
  });
}, 1200);



function showStep(i) {
  var el = STEPS[i].find();
  if (!el) {
    console.warn('[Tour] step ' + i + ' element not found, skipping');
    if (i < STEPS.length - 1) { curr = i + 1; showStep(curr); } else exitTour();
    return;
  }
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setTimeout(function(){ renderStep(el, STEPS[i], i); }, 600);
}

function renderStep(el, step, i) {
  var PAD = 8;
  var r = el.getBoundingClientRect();
  var W = P.innerWidth, H = P.innerHeight;
  // If element is still fully off-screen, scroll hasn't finished — retry in 150ms
  // Note: don't check width===0 as some Streamlit wrappers report zero width
  if (r.bottom < 0 || r.top > H) {
    setTimeout(function(){ renderStep(el, step, i); }, 150);
    return;
  }

  var sp = D.getElementById('tc-spot');
  // Safety: skip elements that are hidden or collapsed to the top-left corner.
  // Hidden Streamlit tab panels can report non-zero width/height but sit at (0,0).
  if ((r.width === 0 && r.height === 0) || (r.top < 40 && r.left < 10 && r.height < 20)) {
    console.warn('[Tour] step element appears hidden or at origin, skipping');
    if (i < STEPS.length - 1) { curr = i + 1; showStep(curr); } else exitTour();
    return;
  }
  sp.style.display = 'block';
  sp.style.top    = (r.top    - PAD) + 'px';
  sp.style.left   = (r.left   - PAD) + 'px';
  sp.style.width  = (r.width  + PAD * 2) + 'px';
  sp.style.height = (r.height + PAD * 2) + 'px';

  var CW = 340, CH = 220;

  // Spotlight boundaries (with padding)
  var spotTop    = r.top    - PAD;
  var spotBottom = r.bottom + PAD;
  var spotLeft   = r.left   - PAD;
  var spotRight  = r.right  + PAD;

  var GAP = 20; // minimum clear gap between card and spotlight
  var top, left;

  // Always anchor horizontally to left edge of element (clamped to viewport)
  left = Math.min(Math.max(r.left, 8), W - CW - 8);

  // Prefer below; fall back to above
  if (spotBottom + GAP + CH <= H - 8) {
    top = spotBottom + GAP;
  } else if (spotTop - GAP - CH >= 8) {
    top = spotTop - GAP - CH;
  } else {
    // Not enough room above or below — use below anyway and accept scroll
    top = spotBottom + GAP;
  }

  // Clamp to viewport
  top  = Math.max(8, Math.min(top,  H - CH - 8));
  left = Math.max(8, Math.min(left, W - CW - 8));

  // Final collision check — if card still overlaps spotlight, push it clear
  var cardBottom = top + CH;
  var cardRight  = left + CW;
  var overlapV = top < spotBottom && cardBottom > spotTop;
  var overlapH = left < spotRight && cardRight  > spotLeft;
  if (overlapV && overlapH) {
    // Push below spotlight if there's any room, else push above
    if (spotBottom + GAP + CH <= H - 8) {
      top = spotBottom + GAP;
    } else {
      top = Math.max(8, spotTop - GAP - CH);
    }
  }

  var dots = STEPS.map(function(_, x){
    return '<div class="tc-dot ' + (x < i ? 'dn' : x === i ? 'on' : '') + '"></div>';
  }).join('');

  var isLast  = (i === STEPS.length - 1);
  var waiting = !!step.waitFor;

  var cd = D.getElementById('tc-card');
  cd.style.display = 'block';
  cd.style.top  = top  + 'px';
  cd.style.left = left + 'px';
  cd.innerHTML =
    '<div class="lbl">Step ' + (i+1) + ' of ' + STEPS.length + '</div>' +
    '<div class="prog">' + dots + '</div>' +
    '<h3>' + step.title + '</h3>' +
    '<p>'  + step.text  + '</p>' +
    (waiting
      ? '<div class="btns"><button class="tb tx" id="tc-x">&#x2715; Exit</button>' +
        '<span style="font-size:13px;color:#93c5fd;margin-left:8px;">' +
          (step.waitMsg || '&#x23F3; Waiting...') +
        '</span></div>'
      : '<div class="btns">' +
          '<button class="tb tx" id="tc-x">&#x2715; Exit</button>' +
          (i > 0 ? '<button class="tb tp" id="tc-p">&#x2190; Back</button>' : '') +
          '<button class="tb tn" id="tc-n">' + (isLast ? '&#x1F389; Done' : 'Next &#x2192;') + '</button>' +
        '</div>');

  D.getElementById('tc-x').onclick = exitTour;
  if (!waiting) {
    D.getElementById('tc-n').onclick = function(){ isLast ? exitTour() : (curr++, showStep(curr)); };
    var pb = D.getElementById('tc-p');
    if (pb) pb.onclick = function(){ curr--; showStep(curr); };
  }

  if (waiting) {
    if (waitPoll) clearInterval(waitPoll);
    waitPoll = setInterval(function() {
      if (step.waitFor()) {
        clearInterval(waitPoll);
        waitPoll = null;
        curr++;
        showStep(curr);
      }
    }, 400);
  }
}

function exitTour() {
  if (waitPoll) { clearInterval(waitPoll); waitPoll = null; }
  D.getElementById('tc-backdrop').style.display = 'none';
  D.getElementById('tc-spot').style.display     = 'none';
  D.getElementById('tc-card').style.display     = 'none';
}
})();
</script>
</body>
</html>
"""
    components.html(_TOUR_HTML, height=0)

    # ── Outer-tab persistence via sessionStorage ──────────────────────────────────
    # apply_team_theme() (model_utils.py) makes a conditional st.markdown() call
    # whose presence depends on whether player_base_team is set. That shifts the
    # widget-position counter, giving the outer st.tabs a new element ID on first
    # player selection — React resets it to tab 0 (Offensive).
    # Fix: JS saves the active outer tab to sessionStorage on every click and
    # immediately restores it after each Streamlit rerun, before the browser paints,
    # so the user never sees a flash back to the Offensive tab.
    _TAB_PERSIST_HTML = """
<!DOCTYPE html>
<html><head><style>body{margin:0;background:transparent;}</style></head>
<body>
<script>
(function(){
  var P = window.parent, D = P.document;
  var KEY = 'nhp_outer_tab';
  var OUTER = ['Offensive','Defensive','Contract Evaluator','Models','Validation'];

  function outerTabBtns() {
    return Array.from(D.querySelectorAll('button[role="tab"]')).filter(function(b){
      return OUTER.indexOf(b.textContent.trim()) >= 0;
    });
  }

  var restoring = false;
  function restore() {
    if (restoring) return;
    var saved = sessionStorage.getItem(KEY);
    if (!saved) return;
    var target = outerTabBtns().find(function(b){ return b.textContent.trim() === saved; });
    if (target && target.getAttribute('aria-selected') !== 'true') {
      restoring = true;
      target.click();
      setTimeout(function(){ restoring = false; }, 600);
    }
  }

  // Save the clicked tab IMMEDIATELY (synchronously) so that when the
  // MutationObserver fires restore() in the same event loop tick, it
  // already sees the correct saved value and does nothing.
  // Using setTimeout here was the bug: restore() fired via rAF before
  // the delayed save ran, read the stale previous-tab value, and clicked
  // back to Offensive — visible as the page jumping before names appeared.
  var tagged = new WeakSet();
  function attachSave() {
    outerTabBtns().forEach(function(b){
      if (!tagged.has(b)) {
        tagged.add(b);
        b.addEventListener('click', function(){
          sessionStorage.setItem(KEY, b.textContent.trim());
        });
      }
    });
  }

  // After each Streamlit rerun, restore the correct tab before the next paint
  var pending = false;
  var obs = new MutationObserver(function(){
    if (pending) return;
    pending = true;
    requestAnimationFrame(function(){
      pending = false;
      attachSave();
      restore();
    });
  });
  obs.observe(D.body, { childList: true, subtree: true });

  // Initial run
  setTimeout(function(){ attachSave(); restore(); }, 600);
})();
</script>
</body></html>
"""
    components.html(_TAB_PERSIST_HTML, height=0)

    # ── Offensive (nested) ────────────────────────────────────────────────────────
    with tab_off:
        st.caption("Search for a forward to see offensive predictions.")
        off_input = st.selectbox("Search for a forward", options=[""] + _fwd_names,
                                 index=0, key="off_player_input", format_func=_fmt_fwd)

        pred = None
        if not off_input:
            st.session_state.pop("active_team", None)      # reset to black when cleared
            st.session_state.pop("player_base_team", None)

        # ── Stable placeholder for error / traded-banner ──────────────────────
        # Routing all conditional pre-tab content through a single st.empty()
        # slot keeps st.tabs() at a fixed position in the widget tree so
        # Streamlit never resets the active sub-tab on player selection.
        _off_above_tabs = st.empty()

        off_t1, off_t2, off_t3 = st.tabs(["Team Fit", "Next Season", "Roster Insertion"])

        if off_input:
            first = predict_player(off_input, df, team_ctx, fit_models, next_models,
                                   player_profiles, has_age)
            if first is None:
                _off_above_tabs.error(f"No forward found matching '{off_input}'.")
            elif first["traded_teams"]:
                _banner_team = st.session_state.get("off_team_override", first["traded_teams"][-1])
                _bg  = get_team_color(_banner_team, "primary")
                _brd = get_team_color(_banner_team, "secondary")
                _r, _g, _b = int(_bg[1:3], 16), int(_bg[3:5], 16), int(_bg[5:7], 16)
                _txt = "#111111" if (0.299*_r + 0.587*_g + 0.114*_b) / 255 > 0.5 else "#ffffff"
                with _off_above_tabs.container():
                    st.markdown(
                        f"""<div style="background:{_bg};border-left:5px solid {_brd};
                            padding:12px 16px;border-radius:6px;color:{_txt};
                            font-size:15px;margin-bottom:8px;">
                            🔁 <strong>{first['matched']}</strong> was traded in {first['seasons'][0]}.
                            Select their current team:
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    override_team = st.radio(
                        "Current team", options=first["traded_teams"],
                        horizontal=True, key="off_team_override"
                    )
                pred = predict_player(off_input, df, team_ctx, fit_models, next_models,
                                      player_profiles, has_age, override_team=override_team)
                if pred:
                    st.session_state["active_team"]        = pred.get("actual_team")
                    st.session_state["player_base_team"]   = pred.get("actual_team")
                    st.session_state["_last_player_source"] = "offensive"
                    update_team_colors(player_team=pred.get("actual_team"))
            else:
                pred = first
            if pred:
                st.session_state["shared_pred"] = pred
                # New offensive player → reset overrides (use per-tab pid to avoid
                # conflicting with the defensive tab's own _def_pid tracker).
                if st.session_state.get("_off_pid") != pred.get("pid"):
                    st.session_state["_off_pid"]            = pred.get("pid")
                    st.session_state["_last_player_source"] = "offensive"
                    st.session_state["_team_override"]      = False
                    for _k in ("insertion_team", "pair_team_sel", "contract_team"):
                        st.session_state.pop(_k, None)
                # Only apply colors when this tab was the most-recently-used one,
                # so it doesn't overwrite the defensive/contract tab's team color.
                if st.session_state.get("_last_player_source") == "offensive":
                    st.session_state["player_base_team"] = pred.get("actual_team")
                    if not st.session_state.get("_team_override"):
                        st.session_state["active_team"] = pred.get("actual_team")
                    update_team_colors(
                        player_team   = pred.get("actual_team"),
                        override_team = st.session_state.get("active_team") if st.session_state.get("_team_override") else None,
                    )

        with off_t1:
            if pred and pred.get("fit_results") is not None:
                seasons_str = " → ".join(str(s) for s in pred["seasons"])
                age_str     = f"  |  Age {pred['age']:.0f}" if pred.get("age") is not None and pd.notna(pred.get("age")) else ""
                # ── Player headshot + header ─────────────────────────────────
                _col_img, _col_hdr = st.columns([1, 8])
                with _col_img:
                    st.markdown(get_player_headshot_html(pred["pid"]), unsafe_allow_html=True)
                with _col_hdr:
                    st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}  |  Seasons: {seasons_str}")
                    st.caption("Predicted performance based on current weighted skill profile across all 32 teams.")

                # ── Grade metrics ─────────────────────────────────────────────
                _fit_row = pred["fit_results"][pred["fit_results"]["is_actual"]]
                _fit_row = _fit_row.iloc[0] if not _fit_row.empty else pred["fit_results"].iloc[0]

                def _off_pct(col, val, ref_df=df):
                    return float((ref_df[col].dropna() < val).mean() * 100)

                def _score_to_grade(s):
                    if s >= 90: return "A"
                    if s >= 75: return "B+"
                    if s >= 50: return "B"
                    if s >= 35: return "C+"
                    if s >= 20: return "C"
                    return "D"

                _pts_pct  = _off_pct("points_per_game",     _fit_row["pred_points_per_game"])
                _gls_pct  = _off_pct("goals_per_game",      _fit_row["pred_goals_per_game"])
                _gs_pct   = _off_pct("game_score_per_game", _fit_row["pred_game_score_per_game"])
                _role_score = (_pts_pct * 0.5 + _gls_pct * 0.3 + _gs_pct * 0.2)

                gc1, gc2, gc3, gc4 = st.columns(4)
                gc1.metric("Points/GP Grade",    _score_to_grade(_pts_pct),  f"Top {100-_pts_pct:.0f}%")
                gc2.metric("Goals/GP Grade",     _score_to_grade(_gls_pct),  f"Top {100-_gls_pct:.0f}%")
                gc3.metric("Game Score Grade",   _score_to_grade(_gs_pct),   f"Top {100-_gs_pct:.0f}%")
                gc4.metric("Overall Grade",      _score_to_grade(_role_score),f"Top {100-_role_score:.0f}%")

                # ── Per-stat percentile breakdown ─────────────────────────────
                st.markdown("#### Category Breakdown")

                def _off_pct_bar(label, val, pct):
                    color = (
                        "#FFD700" if pct >= 90 else
                        "#4a90d9" if pct >= 75 else
                        "#57a85a" if pct >= 50 else
                        "#e8a838" if pct >= 35 else
                        "#c8102e"
                    )
                    st.markdown(
                        f"**{label}** &nbsp; `{val:.3f}` &nbsp; — &nbsp; "
                        f"<span style='color:{color}'>**{pct:.0f}th%**</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(int(pct))

                _profile = player_profiles[pred["pid"]][0] if pred["pid"] in player_profiles else None

                st.markdown("**Production (on actual team)**")
                _off_pct_bar("Points / Game",     _fit_row["pred_points_per_game"],     _pts_pct)
                _off_pct_bar("Goals / Game",      _fit_row["pred_goals_per_game"],      _gls_pct)
                _off_pct_bar("Game Score / Game", _fit_row["pred_game_score_per_game"], _gs_pct)

                if _profile is not None:
                    st.markdown("**Skill Profile**")
                    for _col, _label in [
                        ("finishing_skill_adj",      "Finishing Skill (adj xG)"),
                        ("hd_shot_share",            "High-Danger Shot Share"),
                        ("hd_finishing",             "High-Danger Finishing"),
                        ("primary_assist_share",     "Primary Assist Share"),
                        ("on_target_rate",           "On-Target Rate"),
                        ("pp_icetime_pct",           "PP Ice Time %"),
                    ]:
                        if _col in _profile.index and not pd.isna(_profile[_col]) and _col in df.columns:
                            _v   = float(_profile[_col])
                            _pct = _off_pct(_col, _v)
                            _off_pct_bar(_label, _v, _pct)

                st.plotly_chart(make_bar_chart(pred["fit_results"], pred["matched"], pred["actual_team"],
                                         f"{pred['matched']}  |  Current skill profile  |  Seasons: {seasons_str}"),
                                         use_container_width=True)
                st.markdown("#### Rankings Table")
                display = show_results_table(pred["fit_results"], pred["actual_team"])
                csv = display.drop(columns="_is_actual").to_csv(index_label="rank")
                st.download_button("Download CSV", data=csv,
                                   file_name=f"{pred['matched'].replace(' ','_')}_team_fit.csv",
                                   mime="text/csv")
            else:
                st.info("Search for a forward above to see predictions.")

        with off_t2:
            if pred and pred.get("next_results") is not None:
                age_str = f"  |  Age {pred['age']:.0f} → {pred['age']+1:.0f}" if pred.get("age") is not None and pd.notna(pred.get("age")) else ""
                _col_img2, _col_hdr2 = st.columns([1, 8])
                with _col_img2:
                    st.markdown(get_player_headshot_html(pred["pid"]), unsafe_allow_html=True)
                with _col_hdr2:
                    st.subheader(f"{pred['matched']}  —  {pred['position']}  |  {pred['actual_team']}{age_str}")
                    st.caption("Predicted next-season performance across all 32 teams.")

                # ── Grade metrics ─────────────────────────────────────────────
                _next_row = pred["next_results"][pred["next_results"]["is_actual"]]
                _next_row = _next_row.iloc[0] if not _next_row.empty else pred["next_results"].iloc[0]

                _npts_pct = _off_pct("points_per_game",     _next_row["pred_points_per_game"])
                _ngls_pct = _off_pct("goals_per_game",      _next_row["pred_goals_per_game"])
                _ngs_pct  = _off_pct("game_score_per_game", _next_row["pred_game_score_per_game"])
                _nrole    = (_npts_pct * 0.5 + _ngls_pct * 0.3 + _ngs_pct * 0.2)

                nc1, nc2, nc3, nc4 = st.columns(4)
                nc1.metric("Points/GP Grade",  _score_to_grade(_npts_pct),  f"Top {100-_npts_pct:.0f}%")
                nc2.metric("Goals/GP Grade",   _score_to_grade(_ngls_pct),  f"Top {100-_ngls_pct:.0f}%")
                nc3.metric("Game Score Grade", _score_to_grade(_ngs_pct),   f"Top {100-_ngs_pct:.0f}%")
                nc4.metric("Overall Grade",    _score_to_grade(_nrole),     f"Top {100-_nrole:.0f}%")

                # ── Per-stat percentile breakdown ─────────────────────────────
                st.markdown("#### Category Breakdown")

                st.markdown("**Next-Season Projection (on actual team)**")
                _off_pct_bar("Points / Game",     _next_row["pred_points_per_game"],     _npts_pct)
                _off_pct_bar("Goals / Game",      _next_row["pred_goals_per_game"],      _ngls_pct)
                _off_pct_bar("Game Score / Game", _next_row["pred_game_score_per_game"], _ngs_pct)

                if _profile is not None and has_age and pred.get("age") is not None and pd.notna(pred.get("age")):
                    st.markdown("**Age Trajectory**")
                    _curr_age = pred["age"]
                    _peak_pts = float(_profile.get("career_peak_points_pg", 0) or 0)
                    _pct_of_peak = float(_profile.get("pct_of_peak_points", 0) or 0)
                    if _peak_pts > 0:
                        st.markdown(
                            f"**Career Peak** &nbsp; `{_peak_pts:.3f} pts/gp` &nbsp; — &nbsp; "
                            f"Currently at `{_pct_of_peak*100:.0f}%` of peak",
                            unsafe_allow_html=True,
                        )
                    _slope = float(_profile.get("recent_3yr_points_slope", 0) or 0)
                    _slope_color = "#57a85a" if _slope >= 0 else "#c8102e"
                    _slope_label = "▲ ascending" if _slope > 0.01 else ("▼ declining" if _slope < -0.01 else "→ stable")
                    st.markdown(
                        f"**3-Year Trend** &nbsp; "
                        f"<span style='color:{_slope_color}'>**{_slope_label}**</span> "
                        f"(`{_slope:+.3f}` pts/gp per season)",
                        unsafe_allow_html=True,
                    )

                st.plotly_chart(make_bar_chart(pred["next_results"], pred["matched"], pred["actual_team"],
                                         f"{pred['matched']}  |  Next season forecast"),
                                         use_container_width=True)
                st.markdown("#### Rankings Table")
                display = show_results_table(pred["next_results"], pred["actual_team"])
                csv = display.drop(columns="_is_actual").to_csv(index_label="rank")
                st.download_button("Download CSV", data=csv,
                                   file_name=f"{pred['matched'].replace(' ','_')}_next_season.csv",
                                   mime="text/csv")
            else:
                st.info("Search for a forward above to see predictions.")

        with off_t3:
            st.caption("Select a team to see where the searched player would slot into their active roster.")
            if not pred or pred.get("position") == "D":
                st.info("Search for a forward above to use this tab.")
            else:
                c1, c2 = st.columns([1, 1])
                insertion_team = c1.selectbox(
                    "Select team to insert player into",
                    options=NHL_TEAMS,
                    index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
                    key="insertion_team",
                    on_change=_on_insertion_team_change
                )
                if c2.button("Refresh roster"):
                    fetch_active_team_roster.clear()

                with st.spinner(f"Building {insertion_team} roster with {pred['matched']} inserted..."):
                    insertion_df, insertion_err = build_player_insertion(
                        pred["pid"], insertion_team, df, team_ctx,
                        fit_models, player_profiles, has_age
                    )

                if insertion_err:
                    st.error(insertion_err)
                elif insertion_df is not None and not insertion_df.empty:
                    searched_row = insertion_df[insertion_df["is_searched_player"]].iloc[0]
                    slot   = searched_row["lineup_slot"]
                    rank   = int(searched_row["rank"])
                    color  = searched_row["slot_color"]
                    total  = len(insertion_df)
                    st.markdown(
                        f"<h3 style='color:{color}'>"
                        f"{pred['matched']} projects as a <b>{slot}</b> player on {insertion_team} "
                        f"(rank {rank} of {total} {'forwards' if pred['position'] in ('C','L','R') else 'defensemen'})"
                        f"</h3>",
                        unsafe_allow_html=True
                    )
                    display = insertion_df[[
                        "rank","player_name","position","lineup_slot","pred_points_gp","pred_goals_gp"
                    ]].copy()
                    display.columns = ["Rank","Player","Pos","Line/Pair","Points/GP","Goals/GP"]
                    def _hi_searched(row):
                        m = insertion_df.loc[insertion_df["player_name"]==row["Player"],"is_searched_player"].values
                        if len(m)>0 and m[0]:
                            return [f"background-color:{color}22;font-weight:bold;border-left:3px solid {color}"]*len(row)
                        return [""]*len(row)
                    st.dataframe(display.style.apply(_hi_searched,axis=1),
                                 use_container_width=True,
                                 height=min(50+len(display)*35,600))
                    slot_counts = display["Line/Pair"].value_counts()
                    is_fwd = pred["position"] in ("C","L","R")
                    slot_labels = (["1st Line","2nd Line","3rd Line","4th Line"] if is_fwd
                                   else ["1st Pair","2nd Pair","3rd Pair","3rd Pair (extra)"])
                    st.markdown("**Roster slot breakdown after insertion:**")
                    cols_slots = st.columns(4)
                    for col_s, lbl in zip(cols_slots, slot_labels):
                        col_s.metric(lbl, int(slot_counts.get(lbl,0)))
                    csv = insertion_df.drop(columns=["slot_color"]).to_csv(index=False)
                    st.download_button("Download roster insertion CSV", data=csv,
                                       file_name=f"{pred['matched'].replace(' ','_')}_{insertion_team}_insertion.csv",
                                       mime="text/csv")

    # ── Defensive (nested) ────────────────────────────────────────────────────────
    with tab_def:
        if not def_models_loaded:
            st.warning("Defensive model not loaded. Ensure defensive_dataset.csv is present.")
        else:
            st.caption("Search for a defenseman to see defensive predictions.")
            def_input = st.selectbox("Search for a defenseman", options=[""] + _def_names,
                                     index=0, key="def_player_input", format_func=_fmt_def)

        def_pred_input     = None
        dpred              = None
        _def_colors_kwargs = None   # deferred — called AFTER st.tabs() below
        _def_first         = None   # held so dpred can be cached after st.tabs()

        # ── Stable placeholder for error message ──────────────────────────────
        # Keeps def_t1/def_t2/def_t3 at a fixed widget-tree position.
        # update_team_colors() (→ st.markdown) and st.spinner() are deliberately
        # deferred to AFTER the st.tabs() call so no conditional widget can shift
        # the inner-tabs' element-ID between reruns.
        _def_above_tabs = st.empty()

        if def_models_loaded and def_input:
            _def_first = def_predict_defenseman(
                def_input, def_df, def_team_ctx,
                def_fit_models, def_next_models, def_player_profiles, def_has_age,
                fit_feature_names=def_fit_feature_names,
                next_feature_names=def_next_feature_names,
                season_df=def_df
            )
            if _def_first is None:
                _def_above_tabs.error(f"No defenseman found matching '{def_input}'.")
            else:
                def_pred_input = {
                    "pid":          _def_first["pid"],
                    "matched":      _def_first["matched"],
                    "actual_team":  _def_first["actual_team"],
                    "position":     "D",
                    "seasons":      _def_first["seasons"],
                    "traded_teams": [],
                    "fit_results":  None,
                    "next_results": None,
                    "age":          None,
                }
                st.session_state["shared_pred"] = def_pred_input
                # New defensive player → use own _def_pid tracker so it doesn't
                # interfere with the offensive tab's _off_pid.
                if st.session_state.get("_def_pid") != def_pred_input.get("pid"):
                    st.session_state["_def_pid"]            = def_pred_input.get("pid")
                    st.session_state["_last_player_source"] = "defensive"
                    st.session_state["_team_override"]      = False
                    for _k in ("insertion_team", "pair_team_sel", "contract_team"):
                        st.session_state.pop(_k, None)
                if st.session_state.get("_last_player_source") == "defensive":
                    st.session_state["player_base_team"] = def_pred_input.get("actual_team")
                    if not st.session_state.get("_team_override"):
                        st.session_state["active_team"] = def_pred_input.get("actual_team")
                    # Store args — update_team_colors() is called after st.tabs()
                    _def_colors_kwargs = dict(
                        player_team   = def_pred_input.get("actual_team"),
                        override_team = st.session_state.get("active_team") if st.session_state.get("_team_override") else None,
                    )

                # Cache dpred immediately if already computed; otherwise defer
                _dpred_key = f"dpred_{_def_first['pid']}"
                if _dpred_key in st.session_state:
                    dpred = st.session_state[_dpred_key]
                # (deferred case handled after st.tabs() below)

        # Use def_pred_input as the local pred alias for defensive tab logic
        pred = def_pred_input

        # Load offensive stats for this defenseman
        _d_off_stats, _d_off_df, _d_off_err = load_defensive_offensive_stats() if def_models_loaded else ({}, None, None)
        _pid_off = _d_off_stats.get(pred["pid"], {}) if pred else {}

        # ── Inner tabs — always at the same widget-tree position ──────────────
        def_t1, def_t2, def_t3 = st.tabs(["Team Fit", "Next Season", "Pairing"])

        # ── Post-tabs: emit the deferred CSS update and cache dpred ──────────
        # These render below the tab widget in the DOM (invisible <style> tag +
        # momentary spinner). Placing them here rather than above st.tabs()
        # ensures the inner tabs always have a stable element-ID across reruns.
        if _def_colors_kwargs is not None:
            update_team_colors(**_def_colors_kwargs)
        if _def_first is not None:
            _dpred_key = f"dpred_{_def_first['pid']}"
            if _dpred_key not in st.session_state:
                with st.spinner("Computing defensive predictions..."):
                    st.session_state[_dpred_key] = _def_first
            if dpred is None:
                dpred = st.session_state[_dpred_key]

        with def_t1:
            if not def_models_loaded:
                st.warning("Defensive model not loaded. Ensure defensive_dataset.csv is present.")
            elif not pred:
                st.info("Search for a defenseman above.")
            elif dpred:
                seasons_str = " → ".join(str(s) for s in dpred["seasons"])
                # Classify and grade
                _sample_preds = dpred["fit_results"][dpred["fit_results"]["is_actual"]].iloc[0].to_dict() \
                               if not dpred["fit_results"][dpred["fit_results"]["is_actual"]].empty \
                               else dpred["fit_results"].iloc[0].to_dict()
                # Classify using career profile (actual stats), not team-adjusted predictions
                _profile_dict = dict(def_player_profiles[pred["pid"]][0]) if pred["pid"] in def_player_profiles else _sample_preds
                _def_grade, _def_score, _def_desc, _def_breakdown = grade_defensive_defenseman(_sample_preds, season_def_df=def_df)
                _off_grade, _off_score, _off_desc, _off_breakdown = grade_offensive_defenseman(_pid_off, season_off_df=_d_off_df) if _pid_off else ("—", 0, "", {})
                _d_type, _d_desc = classify_defenseman_type(_profile_dict, def_score=_def_score, off_score=_off_score)

                _dc_img, _dc_hdr = st.columns([1, 8])
                with _dc_img:
                    st.markdown(get_player_headshot_html(pred["pid"]), unsafe_allow_html=True)
                with _dc_hdr:
                    st.subheader(f"{dpred['matched']}  —  {_d_type}  |  {dpred['actual_team']}  |  Seasons: {seasons_str}")
                    st.caption(_d_desc)

                # Grade display
                def _score_to_grade(s):
                    if s >= 90: return "A"
                    if s >= 75: return "B+"
                    if s >= 50: return "B"
                    if s >= 35: return "C+"
                    if s >= 20: return "C"
                    return "D"
                _combined_num = _def_score * 0.5 + _off_score * 0.5

                if _d_type == "Two-Way D":
                    gc1, gc2, gc3 = st.columns(3)
                    gc1.metric("Defensive Grade", _def_grade, f"Top {100-_def_score:.0f}%")
                    gc2.metric("Offensive Grade", _off_grade, f"Top {100-_off_score:.0f}%")
                    gc3.metric("Combined Grade", _score_to_grade(_combined_num), f"Top {100-_combined_num:.0f}%")
                elif _d_type == "Offensive D":
                    gc1, gc2, gc3 = st.columns(3)
                    gc1.metric("Offensive Grade", _off_grade, f"Top {100-_off_score:.0f}%")
                    gc2.metric("Defensive Grade", _def_grade, f"Top {100-_def_score:.0f}%")
                    gc3.metric("Combined Grade", _score_to_grade(_combined_num), f"Top {100-_combined_num:.0f}%")
                else:  # Defensive D
                    gc1, gc2, gc3 = st.columns(3)
                    gc1.metric("Defensive Grade", _def_grade, f"Top {100-_def_score:.0f}%")
                    gc2.metric("Offensive Grade", _off_grade if _pid_off else "—", f"Top {100-_off_score:.0f}%" if _pid_off else "")
                    gc3.metric("Combined Grade", _score_to_grade(_combined_num), f"Top {100-_combined_num:.0f}%")

                # ── Per-category percentile breakdown ─────────────────────────────
                st.markdown("#### Category Breakdown")

                def _pct_bar(label, val, pct, lower_is_better=False):
                    # For lower-is-better stats, a high percentile = elite suppression
                    # so colors stay the same (high pct = gold), but we show the raw
                    # value context differently so it reads naturally
                    color = (
                        "#FFD700" if pct >= 90 else
                        "#4a90d9" if pct >= 75 else
                        "#57a85a" if pct >= 50 else
                        "#e8a838" if pct >= 35 else
                        "#c8102e"
                    )
                    arrow = "↓ lower is better" if lower_is_better else ""
                    # For lower-is-better, show "Top X%" framing so p10 xGA reads as elite
                    if lower_is_better:
                        pct_label = f"Top {100 - pct:.0f}%" if pct <= 90 else "Elite"
                    else:
                        pct_label = f"{pct:.0f}th%"
                    st.markdown(
                        f"**{label}** &nbsp; `{val}` &nbsp; — &nbsp; "
                        f"<span style='color:{color}'>**{pct_label}**</span> "
                        f"<span style='color:#888;font-size:0.8em'>{arrow}</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(int(pct))

                lower_is_better_cats = {"xGA/60 (5v5)", "PIM/GP"}

                # Defensive breakdown — always shown
                st.markdown("**Defensive**")
                for cat, (val, pct) in _def_breakdown.items():
                    _pct_bar(cat, val, pct, lower_is_better=(cat in lower_is_better_cats))

                # Offensive breakdown — always shown when data available
                if _pid_off and _off_breakdown:
                    st.markdown("**Offensive**")
                    for cat, (val, pct) in _off_breakdown.items():
                        _pct_bar(cat, val, pct)

                st.markdown("#### Rankings Table")
                display = def_show_results_table(dpred["fit_results"], dpred["actual_team"])
                csv = display.drop(columns="_is_actual").to_csv(index_label="rank")
                st.download_button("Download CSV", data=csv,
                                   file_name=f"{dpred['matched'].replace(' ','_')}_def_fit.csv",
                                   mime="text/csv")

        with def_t2:
            if not def_models_loaded:
                st.warning("Defensive model not loaded.")
            elif not pred:
                st.info("Search for a defenseman above.")
            elif dpred:
                _d2_seasons_str = " → ".join(str(s) for s in dpred["seasons"])
                _d2_age         = dpred.get("age")
                _d2_age_str     = (
                    f"  |  Age {_d2_age:.0f} → {_d2_age+1:.0f}"
                    if _d2_age is not None and pd.notna(_d2_age) else ""
                )

                # ── Player headshot + header ─────────────────────────────────
                _d2_img, _d2_hdr = st.columns([1, 8])
                with _d2_img:
                    st.markdown(get_player_headshot_html(pred["pid"]), unsafe_allow_html=True)
                with _d2_hdr:
                    st.subheader(
                        f"{dpred['matched']}  —  D  |  {dpred['actual_team']}{_d2_age_str}"
                    )
                    st.caption("Next-season defensive forecast based on current profile and trajectory.")

                # ── Grade helpers (local copies so def_t2 is self-contained) ─
                def _d2_score_to_grade(s):
                    if s >= 90: return "A"
                    if s >= 75: return "B+"
                    if s >= 50: return "B"
                    if s >= 35: return "C+"
                    if s >= 20: return "C"
                    return "D"

                def _d2_pct_bar(label, val, pct, lower_is_better=False):
                    color = (
                        "#FFD700" if pct >= 90 else
                        "#4a90d9" if pct >= 75 else
                        "#57a85a" if pct >= 50 else
                        "#e8a838" if pct >= 35 else
                        "#c8102e"
                    )
                    arrow = "↓ lower is better" if lower_is_better else ""
                    if lower_is_better:
                        pct_label = f"Top {100-pct:.0f}%" if pct <= 90 else "Elite"
                    else:
                        pct_label = f"{pct:.0f}th%"
                    st.markdown(
                        f"**{label}** &nbsp; `{val}` &nbsp; — &nbsp; "
                        f"<span style='color:{color}'>**{pct_label}**</span> "
                        f"<span style='color:#888;font-size:0.8em'>{arrow}</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(int(pct))

                _d2_lib_cats = {"xGA/60 (5v5)", "PIM/GP"}

                # ── Compute grades from next_results ─────────────────────────
                _d2_next_is_actual = dpred["next_results"][dpred["next_results"]["is_actual"]]
                _d2_next_sample = (
                    _d2_next_is_actual.iloc[0].to_dict()
                    if not _d2_next_is_actual.empty
                    else dpred["next_results"].iloc[0].to_dict()
                )
                _d2_def_grade, _d2_def_score, _, _d2_def_bkdn = grade_defensive_defenseman(
                    _d2_next_sample, season_def_df=def_df
                )
                _d2_off_grade, _d2_off_score, _, _d2_off_bkdn = (
                    grade_offensive_defenseman(_pid_off, season_off_df=_d_off_df)
                    if _pid_off else ("—", 0, "", {})
                )
                _d2_combined = _d2_def_score * 0.5 + _d2_off_score * 0.5

                # ── Grade metrics ─────────────────────────────────────────────
                _d2_type, _ = classify_defenseman_type(
                    dict(def_player_profiles[pred["pid"]][0])
                    if pred["pid"] in def_player_profiles else _d2_next_sample,
                    def_score=_d2_def_score, off_score=_d2_off_score,
                )
                if _d2_type == "Offensive D":
                    nd1, nd2, nd3 = st.columns(3)
                    nd1.metric("Offensive Grade",  _d2_off_grade,                        f"Top {100-_d2_off_score:.0f}%")
                    nd2.metric("Defensive Grade",  _d2_def_grade,                        f"Top {100-_d2_def_score:.0f}%")
                    nd3.metric("Combined Grade",   _d2_score_to_grade(_d2_combined),     f"Top {100-_d2_combined:.0f}%")
                else:  # Defensive D or Two-Way D
                    nd1, nd2, nd3 = st.columns(3)
                    nd1.metric("Defensive Grade",  _d2_def_grade,                        f"Top {100-_d2_def_score:.0f}%")
                    nd2.metric("Offensive Grade",  _d2_off_grade if _pid_off else "—",   f"Top {100-_d2_off_score:.0f}%" if _pid_off else "")
                    nd3.metric("Combined Grade",   _d2_score_to_grade(_d2_combined),     f"Top {100-_d2_combined:.0f}%")

                # ── Per-category percentile breakdown ─────────────────────────
                st.markdown("#### Category Breakdown")

                st.markdown("**Next-Season Defensive Projection**")
                for cat, (val, pct) in _d2_def_bkdn.items():
                    _d2_pct_bar(cat, val, pct, lower_is_better=(cat in _d2_lib_cats))

                if _pid_off and _d2_off_bkdn:
                    st.markdown("**Offensive**")
                    for cat, (val, pct) in _d2_off_bkdn.items():
                        _d2_pct_bar(cat, val, pct)

                # ── Age trajectory ────────────────────────────────────────────
                _d2_profile = def_player_profiles.get(pred["pid"])
                if (
                    _d2_profile is not None
                    and def_has_age
                    and _d2_age is not None
                    and pd.notna(_d2_age)
                ):
                    _d2_p = _d2_profile[0]
                    st.markdown("**Age Trajectory**")
                    _d2_peak = float(_d2_p.get("career_peak_hits_pg", 0) or 0)
                    _d2_pct_peak = float(_d2_p.get("pct_of_peak_hits", 0) or 0)
                    if _d2_peak > 0:
                        st.markdown(
                            f"**Career Peak (Hits/GP)** &nbsp; `{_d2_peak:.3f}` &nbsp; — &nbsp; "
                            f"Currently at `{_d2_pct_peak*100:.0f}%` of peak",
                            unsafe_allow_html=True,
                        )
                    _d2_slope = float(_d2_p.get("recent_3yr_hits_slope", 0) or 0)
                    _d2_slope_color = "#57a85a" if _d2_slope >= 0 else "#c8102e"
                    _d2_slope_label = (
                        "▲ ascending" if _d2_slope > 0.01
                        else ("▼ declining" if _d2_slope < -0.01 else "→ stable")
                    )
                    st.markdown(
                        f"**3-Year Trend** &nbsp; "
                        f"<span style='color:{_d2_slope_color}'>**{_d2_slope_label}**</span> "
                        f"(`{_d2_slope:+.3f}` hits/gp per season)",
                        unsafe_allow_html=True,
                    )

                # ── Rankings Table ────────────────────────────────────────────
                st.markdown("#### Rankings Table")
                display = def_show_results_table(dpred["next_results"], dpred["actual_team"])
                csv = display.drop(columns="_is_actual").to_csv(index_label="rank")
                st.download_button("Download CSV", data=csv,
                                   file_name=f"{dpred['matched'].replace(' ','_')}_def_next.csv",
                                   mime="text/csv")

        with def_t3:
            if not def_models_loaded:
                st.warning("Defensive model not loaded.")
            elif not pred:
                st.info("Search for a defenseman above.")
            elif dpred:
                pc1, pc2 = st.columns([1, 1])
                pair_team = pc1.selectbox(
                    "Select team",
                    options=NHL_TEAMS,
                    index=NHL_TEAMS.index(dpred["actual_team"]) if dpred["actual_team"] in NHL_TEAMS else 0,
                    key="pair_team_sel",
                    on_change=_on_pair_team_change
                )
                n_games = st.slider(
                    "Games to include in pairing data",
                    min_value=10, max_value=82, value=25, step=5,
                    help="More games = more stable pairs but slower load. 25 games reflects recent pairings well.",
                    key="pair_games_slider"
                )

                # ── Fetch shift pairs with live progress bar ──────────────────
                _pair_cache_key = f"_pairs_{pair_team}_{n_games}"

                _do_refresh = pc2.button("Refresh roster & shifts", key="pair_refresh")
                if _do_refresh:
                    def_fetch_team_roster_d.clear()
                    fetch_actual_pairs.clear()
                    st.session_state.pop(_pair_cache_key, None)

                if _pair_cache_key not in st.session_state:
                    _prog_label = st.empty()
                    _prog_bar   = st.progress(0)

                    def _on_game_progress(done, total):
                        pct = int(done / total * 100)
                        _prog_label.markdown(
                            f"<small style='color:#aaa'>Fetching {pair_team} shifts — "
                            f"game {done} of {total}</small>",
                            unsafe_allow_html=True,
                        )
                        _prog_bar.progress(pct)

                    _fetched = stream_fetch_actual_pairs(
                        pair_team,
                        d_pids=None,
                        n_games=n_games,
                        on_progress=_on_game_progress,
                    )
                    st.session_state[_pair_cache_key] = _fetched
                    _prog_bar.empty()
                    _prog_label.empty()

                _prefetched = st.session_state[_pair_cache_key]

                with st.spinner("Building pairing model..."):
                    pair_result = def_build_pairing_insertion(
                        dpred["pid"], pair_team, def_df, def_team_ctx,
                        def_fit_models, def_player_profiles, def_has_age,
                        feature_names=def_fit_feature_names,
                        n_games=n_games,
                        _prefetched_pairs=_prefetched,
                    )

                if isinstance(pair_result, tuple) and len(pair_result) == 6:
                    depth_pairs, scratched, player_scores, cascade_log, unmodeled, insertion = pair_result

                    if not depth_pairs and not player_scores:
                        st.error(insertion.get("pair_err", "Could not build pairings. Try refreshing."))
                    else:
                        searched_name  = dpred["matched"]
                        searched_score = insertion["searched_score"]
                        searched_info  = player_scores.get(dpred["pid"], {})
                        searched_type  = searched_info.get("d_type", "—")
                        partner_name   = insertion["partner_name"]
                        partner_slot   = insertion["partner_slot"]
                        pair_err       = insertion.get("pair_err")

                        if pair_err:
                            st.caption(f"Note: Could not fetch shift data ({pair_err}). Using model-ranked order.")

                        # ── Player header ──────────────────────────────────────────────
                        st.markdown(f"### {searched_name} — {searched_type} | Combined Score: {searched_score:.0f}")
                        if partner_name != "—":
                            st.success(f"Projected pair: **{searched_name}** with **{partner_name}** ({partner_slot})")

                        # ── Depth chart after insertion ────────────────────────────────
                        st.divider()
                        is_returning = dpred["pid"] in {p["player_id"] for p in (def_fetch_team_roster_d(pair_team) or [])}
                        chart_title  = f"#### {pair_team} Defensive Depth Chart — Current Pairs" if is_returning else f"#### {pair_team} Defensive Depth Chart — After Insertion"
                        chart_caption = ("Gold = highlighted player. Season-long shift pairs shown as-is." if is_returning
                                         else "Gold = new player. Pairs anchored from season-long shift data. Cascade ripples down — weakest player scratched.")
                        st.markdown(chart_title)
                        st.caption(chart_caption)

                        SLOT_COLORS_PAIRS = {
                            "1st Pair": "#FFD700",
                            "2nd Pair": "#4a90d9",
                            "3rd Pair": "#57a85a",
                            "4th Pair": "#888888",
                        }

                        for pair in depth_pairs:
                            slot_label = pair["slot"]
                            color = SLOT_COLORS_PAIRS.get(slot_label, "#888888")
                            p1 = player_scores.get(pair["pid1"], {})
                            p2 = player_scores.get(pair["pid2"], {})
                            is_new1 = pair["pid1"] == dpred["pid"]
                            is_new2 = pair["pid2"] == dpred["pid"]
                            type1   = p1.get("d_type", "")
                            type2   = p2.get("d_type", "")
                            hand1   = pair.get("shoots1", "")
                            hand2   = pair.get("shoots2", "")
                            hand_ok = pair.get("hand_match", False)
                            hand_icon = "🤝" if hand_ok else ("⚠️" if (hand1 and hand2) else "")

                            col_slot, col_p1, col_vs, col_p2, col_score = st.columns([1.2, 2.5, 0.3, 2.5, 1.0])
                            col_slot.markdown(
                                f"<span style='color:{color};font-weight:bold'>{slot_label}</span>"
                                f"<br><small>{hand_icon}</small>",
                                unsafe_allow_html=True
                            )
                            p1_style = "color:#FFD700;font-weight:bold" if is_new1 else ""
                            p2_style = "color:#FFD700;font-weight:bold" if is_new2 else ""
                            hand_label1 = f" ({hand1})" if hand1 else ""
                            hand_label2 = f" ({hand2})" if hand2 else ""
                            col_p1.markdown(
                                f"<span style='{p1_style}'>**{pair['name1']}**{hand_label1} ({pair['score1']:.0f})</span>  \n*{type1}*",
                                unsafe_allow_html=True
                            )
                            col_vs.markdown("—")
                            col_p2.markdown(
                                f"<span style='{p2_style}'>**{pair['name2']}**{hand_label2} ({pair['score2']:.0f})</span>  \n*{type2}*",
                                unsafe_allow_html=True
                            )
                            col_score.markdown(f"Avg: **{pair['pair_score']:.0f}**")

                        st.caption("Score = combined grade (defensive + offensive weighted by player type).")

                        # ── Scratched players ──────────────────────────────────────────
                        if scratched:
                            st.divider()
                            st.markdown("#### Scratched / Excess D-men")
                            scratch_cols = st.columns(min(len(scratched), 4))
                            for col, pid in zip(scratch_cols, scratched):
                                info = player_scores.get(pid, {})
                                col.metric(
                                    info.get("player_name", str(pid)),
                                    f"Score: {info.get('combined_score', 0):.0f}",
                                    info.get("d_type", ""),
                                )

                        # ── Cascade log ────────────────────────────────────────────────
                        if cascade_log:
                            st.divider()
                            st.markdown("#### Displacement Cascade")
                            st.caption("Step-by-step ripple: each displaced player finds their next best slot.")
                            for i, entry in enumerate(cascade_log):
                                action = entry.get("action", "")
                                player = entry.get("player", "")
                                slot   = entry.get("slot", "—")
                                disp   = entry.get("displaced", "—")
                                is_new = player == searched_name

                                if action == "scratched":
                                    st.markdown(
                                        f"<span style='color:#c8102e'>✗</span> "
                                        f"**{player}** could not improve any pair → **Scratched**",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    color = "#FFD700" if is_new else "#4a90d9"
                                    disp_str = f" (displaces {disp})" if disp != "—" else ""
                                    st.markdown(
                                        f"<span style='color:{color}'>↓</span> "
                                        f"**{player}** → **{slot}**{disp_str}",
                                        unsafe_allow_html=True
                                    )

                        # ── Players not in model ───────────────────────────────────────
                        if unmodeled:
                            with st.expander(f"{len(unmodeled)} rostered D-men not in model data"):
                                for pid in unmodeled:
                                    name = def_player_profiles.get(pid, ({"player_name": str(pid)},))[0].get("player_name", str(pid))
                                    st.caption(f"• {name} — not enough historical seasons in training data")

                        # ── Full scores table ──────────────────────────────────────────
                        with st.expander("Full model scores for all D-men"):
                            rows_table = []
                            for pid, info in player_scores.items():
                                rows_table.append({
                                    "Player":    info["player_name"],
                                    "Type":      info.get("d_type", "—"),
                                    "Combined":  info.get("combined_score", info["defensive_score"]),
                                    "Def Score": info["defensive_score"],
                                    "Hits/GP":   round(info.get("ind_hits_pg", 0), 2),
                                    "TK/GP":     round(info.get("ind_takeaways_pg", 0), 3),
                                    "xGA/60":    round(info.get("xg_against_per60_5v5", 0), 3),
                                    "PIM/GP":    round(info.get("pim_pg", 0), 2),
                                    "Is New Player": info.get("is_searched_player", False),
                                })
                            scores_df   = pd.DataFrame(rows_table).sort_values("Combined", ascending=False).reset_index(drop=True)
                            is_new_mask = scores_df["Is New Player"].values
                            display_df  = scores_df.drop(columns="Is New Player")

                            def _hi_new(row):
                                return (
                                    ["background-color:#FFD70022;font-weight:bold"] * len(row)
                                    if is_new_mask[row.name] else [""] * len(row)
                                )

                            st.dataframe(display_df.style.apply(_hi_new, axis=1),
                                         use_container_width=True, hide_index=True)
                            st.download_button("Download pairing CSV",
                                               data=pd.DataFrame(rows_table).to_csv(index=False),
                                               file_name=f"{dpred['matched'].replace(' ','_')}_{pair_team}_pairing.csv",
                                               mime="text/csv")
                else:
                    st.error("Unexpected result from pairing function. Try refreshing.")

    # ── Contract Evaluator ────────────────────────────────────────────────────────
    with tab_contract:
        st.subheader("Contract Evaluator")
        st.caption(
            "Projects a player's production across multiple seasons using empirical age curves. "
            "Works for both forwards (offensive stats) and defensemen (defensive stats). "
            "Confidence decreases in later years — use ranges rather than exact numbers."
        )

        contract_input = st.selectbox("Search for a player", options=[""] + _all_names,
                                      index=0, key="contract_player_input", format_func=_fmt_all)
        pred = None
        if contract_input:
            first_c = predict_player(contract_input, df, team_ctx, fit_models, next_models,
                                     player_profiles, has_age)
            if first_c is not None:
                if first_c["traded_teams"]:
                    _banner_team_c = st.session_state.get("contract_team_override", first_c["traded_teams"][-1])
                    _bg  = get_team_color(_banner_team_c, "primary")
                    _brd = get_team_color(_banner_team_c, "secondary")
                    _r, _g, _b = int(_bg[1:3], 16), int(_bg[3:5], 16), int(_bg[5:7], 16)
                    _txt = "#111111" if (0.299*_r + 0.587*_g + 0.114*_b) / 255 > 0.5 else "#ffffff"
                    st.markdown(
                        f"""<div style="background:{_bg};border-left:5px solid {_brd};
                            padding:12px 16px;border-radius:6px;color:{_txt};
                            font-size:15px;margin-bottom:8px;">
                            🔁 <strong>{first_c['matched']}</strong> was traded. Select current team:
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    ov = st.radio("Current team", options=first_c["traded_teams"],
                                  horizontal=True, key="contract_team_override")
                    pred = predict_player(contract_input, df, team_ctx, fit_models, next_models,
                                         player_profiles, has_age, override_team=ov)
                else:
                    pred = first_c
                if pred:
                    if st.session_state.get("_contract_pid") != pred.get("pid"):
                        st.session_state["_contract_pid"]       = pred.get("pid")
                        st.session_state["_last_player_source"] = "contract"
                        st.session_state["_team_override"]      = False
                        for _k in ("insertion_team", "pair_team_sel", "contract_team"):
                            st.session_state.pop(_k, None)
                    if st.session_state.get("_last_player_source") == "contract":
                        st.session_state["player_base_team"] = pred.get("actual_team")
                        if not st.session_state.get("_team_override"):
                            st.session_state["active_team"] = pred.get("actual_team")
                        # The JS watcher only uses --team-bg-override for Roster
                        # Insertion / Pairing inner tabs. Contract Evaluator always
                        # falls back to --team-bg-base, so we put the display team
                        # (signing team when overridden, actual team otherwise)
                        # into player_team so it lands in --team-bg-base.
                        _ct_bg = st.session_state.get("active_team", pred.get("actual_team"))
                        update_team_colors(player_team=_ct_bg, override_team=_ct_bg)
            elif def_models_loaded:
                def_c = def_predict_defenseman(
                    contract_input, def_df, def_team_ctx,
                    def_fit_models, def_next_models, def_player_profiles, def_has_age,
                    fit_feature_names=def_fit_feature_names,
                    next_feature_names=def_next_feature_names,
                    season_df=def_df
                )
                if def_c is not None:
                    pred = {
                        "pid": def_c["pid"], "matched": def_c["matched"],
                        "actual_team": def_c["actual_team"], "position": "D",
                        "seasons": def_c["seasons"], "traded_teams": [],
                        "fit_results": None, "next_results": None, "age": None,
                    }
                    # Mirror the forward branch: update session state and team
                    # background for defensemen so the page background loads.
                    if st.session_state.get("_contract_pid") != pred.get("pid"):
                        st.session_state["_contract_pid"]       = pred.get("pid")
                        st.session_state["_last_player_source"] = "contract"
                        st.session_state["_team_override"]      = False
                        for _k in ("insertion_team", "pair_team_sel", "contract_team"):
                            st.session_state.pop(_k, None)
                    if st.session_state.get("_last_player_source") == "contract":
                        st.session_state["player_base_team"] = pred.get("actual_team")
                        if not st.session_state.get("_team_override"):
                            st.session_state["active_team"] = pred.get("actual_team")
                        _ct_bg = st.session_state.get("active_team", pred.get("actual_team"))
                        update_team_colors(player_team=_ct_bg, override_team=_ct_bg)
                else:
                    st.error(f"No player found matching '{contract_input}'.")
            else:
                st.error(f"No player found matching '{contract_input}'.")

        if not pred:
            st.info("Search for a player above to use the contract evaluator.")
        else:
            is_d_contract = pred.get("position") == "D"

            # ── Controls ───────────────────────────────────────────────────────────
            cc1, cc2, cc3 = st.columns(3)
            contract_team = cc1.selectbox(
                "Team signing the player",
                options=NHL_TEAMS,
                index=NHL_TEAMS.index(pred["actual_team"]) if pred["actual_team"] in NHL_TEAMS else 0,
                key="contract_team",
                on_change=_on_contract_team_change
            )
            # Resolve current age — always try ages CSV first so season offset is applied
            _pid = pred.get("pid")
            curr_age = None

            if os.path.exists(AGES_FILE):
                try:
                    _ages_df = pd.read_csv(AGES_FILE)
                    _row = _ages_df[_ages_df["player_id"] == _pid].sort_values("season", ascending=False)
                    if not _row.empty and pd.notna(_row.iloc[0].get("age")):
                        _latest_season = int(_row.iloc[0]["season"])
                        _base_age = float(_row.iloc[0]["age"])
                        # Adjust forward to current season (2026)
                        curr_age = _base_age + max(0, 2026 - _latest_season)
                except Exception:
                    pass

            if not curr_age and is_d_contract and def_models_loaded:
                _prof = def_player_profiles.get(_pid, (None,))[0]
                if _prof is not None:
                    curr_age = _prof.get("age")

            if not curr_age:
                curr_age = pred.get("age")

            curr_age = float(curr_age) if curr_age and not (isinstance(curr_age, float) and curr_age != curr_age) else 28.0

            # CBA limits based on signing team
            cba = get_cba_limits(curr_age, pred["actual_team"], contract_team)

            n_years = cc2.slider(
                "Contract length (years)",
                min_value=1,
                max_value=cba["max_years"],
                value=min(cba["recommended"], cba["max_years"]),
            )
            cc3.metric("Current Age", f"{curr_age:.0f}")

            # ── CBA Info bar ───────────────────────────────────────────────────────
            cba_cols = st.columns(4)
            signing_type = "Re-signing (same team)" if cba["is_same_team"] else "New signing (different team)"
            cba_cols[0].metric("Signing Type",    signing_type)
            cba_cols[1].metric("CBA Max Length",  f"{cba['max_years']} years")
            cba_cols[2].metric("Recommended Max", f"{cba['recommended']} years")
            cba_cols[3].metric("Age at Expiry",   f"{curr_age + n_years:.0f}")

            # 35+ rule warning
            if cba["is_35_signing"]:
                st.error(
                    f"35+ Rule: {pred['matched']} is {curr_age:.0f} at signing. "
                    "The cap hit counts against your team even if the player retires early. "
                    "This creates significant cap recapture risk."
                )
            elif cba["hits_35_rule"]:
                st.warning(
                    f"35+ Rule alert: {pred['matched']} will reach age 35 during this contract. "
                    "If they retire before expiry, the cap hit remains on your books."
                )

            if n_years > cba["recommended"]:
                st.warning(
                    f"This contract is longer than the recommended maximum of {cba['recommended']} years "
                    f"based on the player's age curve. Years {cba['recommended']+1}+ carry high uncertainty."
                )

            # ── Run projection ─────────────────────────────────────────────────────
            # Resolve dpred from session state (set by Defensive tab if a D-man was searched)
            _contract_dpred = None
            if is_d_contract and pred:
                _contract_dpred_key = f"dpred_{pred['pid']}"
                _contract_dpred = st.session_state.get(_contract_dpred_key)
                if _contract_dpred is None:
                    with st.spinner("Loading defensive predictions for contract..."):
                        _contract_dpred = def_predict_defenseman(
                            pred["matched"], def_df, def_team_ctx,
                            def_fit_models, def_next_models, def_player_profiles, def_has_age,
                            fit_feature_names=def_fit_feature_names,
                            next_feature_names=def_next_feature_names,
                            season_df=def_df
                        )
                        if _contract_dpred:
                            st.session_state[_contract_dpred_key] = _contract_dpred

            with st.spinner("Projecting contract years..."):
                proj_rows, proj_err = build_contract_projection(
                    pred["matched"], pred, _contract_dpred if is_d_contract else None,
                    df, team_ctx, fit_models, next_models, player_profiles, has_age,
                    def_df if def_models_loaded else pd.DataFrame(),
                    def_team_ctx if def_models_loaded else pd.DataFrame(),
                    def_fit_models if def_models_loaded else {},
                    def_player_profiles if def_models_loaded else {},
                    def_has_age if def_models_loaded else False,
                    contract_team, n_years,
                    def_fit_feature_names=def_fit_feature_names if def_models_loaded else None,
                    curr_age=curr_age
                )

            if proj_err:
                st.error(proj_err)
            elif proj_rows:
                risk_label, risk_color, risk_explanation = contract_risk_rating(proj_rows, is_d_contract)

                # ── Risk header ────────────────────────────────────────────────────
                st.markdown(
                    f"<h3>{pred['matched']} on {contract_team} — "
                    f"<span style='color:{risk_color}'>{risk_label}</span></h3>",
                    unsafe_allow_html=True
                )
                if risk_explanation:
                    st.caption(risk_explanation)

                # ── Year-by-year table ─────────────────────────────────────────────
                st.markdown("#### Year-by-Year Projection")
                st.caption("Confidence reflects uncertainty compounding over time. Use wider mental ranges in later years.")

                if is_d_contract:
                    proj_df = pd.DataFrame([{
                        "Year":           f"Year {r['year']} (Age {r['age']:.0f})",
                        "Hits/GP":        r["hits_pg"],
                        "Takeaways/GP":   r["takeaways_pg"],
                        "xGA/60 (5v5)":   r["goals_against_pg"],
                        "PIM/GP":         r["pim_pg"],
                        "Def %ile":       f"Top {100 - r['def_score']:.0f}%" if r['def_score'] > 10 else "Elite",
                        "Off %ile":       f"Top {100 - r['off_score']:.0f}%" if r['off_score'] > 10 else "Elite",
                        "Confidence":     f"{r['confidence']*100:.0f}%",
                    } for r in proj_rows])
                else:
                    proj_df = pd.DataFrame([{
                        "Year":       f"Year {r['year']} (Age {r['age']:.0f})",
                        "Points/GP":  r["points_pg"],
                        "Goals/GP":   r["goals_pg"],
                        "Pts %ile":   f"Top {100 - r['pts_pct']:.0f}%" if r.get('pts_pct', 50) > 10 else "Elite",
                        "Confidence": f"{r['confidence']*100:.0f}%",
                    } for r in proj_rows])

                st.dataframe(proj_df, use_container_width=True, hide_index=True)

                # ── Trend chart ────────────────────────────────────────────────────
                st.markdown("#### Production Trend")
                fig_c, ax_c = plt.subplots(figsize=(10, 4))
                fig_c.patch.set_facecolor("#0e1117")
                ax_c.set_facecolor("#0e1117")

                years  = [r["year"] for r in proj_rows]
                labels = [f"Yr {r['year']} (Age {r['age']:.0f})" for r in proj_rows]

                if is_d_contract:
                    vals   = [r["def_score"] for r in proj_rows]
                    ylabel = "Defensive Percentile"
                    color  = "#4a90d9"
                else:
                    vals   = [r.get("pts_pct", 50) for r in proj_rows]
                    ylabel = "Points Percentile"
                    color  = "#4a90d9"

                confs = [r["confidence"] for r in proj_rows]

                # Plot line with confidence band
                ax_c.plot(years, vals, color=color, linewidth=2.5, marker="o", markersize=8, zorder=3)
                # Asymmetric confidence band — uncertainty is mostly downside risk.
                # A player is unlikely to dramatically improve late in a contract,
                # but could decline. Upside = 25% of spread, downside = 75%.
                spread = [v * (1 - c) * 0.4 for v, c in zip(vals, confs)]
                upper  = [v + s * 0.25 for v, s in zip(vals, spread)]
                lower  = [max(v - s * 0.75, 0) for v, s in zip(vals, spread)]
                ax_c.fill_between(years, lower, upper,
                                  alpha=0.2, color=color, label="Confidence range")
                ax_c.set_xticks(years)
                ax_c.set_xticklabels(labels, color="white", fontsize=9)
                ax_c.set_ylabel(ylabel, color="white", fontsize=10)
                ax_c.tick_params(colors="white")
                ax_c.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
                for spine in ax_c.spines.values():
                    spine.set_edgecolor("#333")
                plt.tight_layout()
                st.pyplot(fig_c)
                plt.close()

                # ── Summary recommendation ─────────────────────────────────────────
                st.divider()
                st.markdown("#### Contract Recommendation")
                yr1     = proj_rows[0]
                yr_last = proj_rows[-1]

                if is_d_contract:
                    y1_score    = yr1["def_score"]
                    yn_score    = yr_last["def_score"]
                    decline     = max(y1_score - yn_score, 0)
                    st.markdown(
                        f"- **Year 1 Defensive Percentile:** {y1_score:.0f}th% among all D-men  \n"
                        f"- **Year {n_years} Defensive Percentile:** {yn_score:.0f}th%  \n"
                        f"- **Projected percentile drop:** {decline:.0f} points  \n"
                        f"- **Risk rating:** {risk_label}"
                    )
                else:
                    y1_pts    = yr1["points_pg"]
                    yn_pts    = yr_last["points_pg"]
                    total_pts = sum(r["points_pg"] * 82 for r in proj_rows)
                    st.markdown(
                        f"- **Year 1 Points/GP:** {y1_pts:.3f}  \n"
                        f"- **Year {n_years} Points/GP:** {yn_pts:.3f}  \n"
                        f"- **Projected total points:** ~{total_pts:.0f} over {n_years} years  \n"
                        f"- **Risk rating:** {risk_label}"
                    )

                # CBA recommendation box
                st.markdown("**CBA Summary:**")
                rec_col1, rec_col2 = st.columns(2)
                signing_lbl = "Same team (re-signing)" if cba["is_same_team"] else "New team"
                rec_col1.info(
                    f"Same-team max: **7 years**  \n"
                    f"New-team max: **6 years**  \n"
                    f"Signing type: **{signing_lbl}**  \n"
                    f"CBA max for this deal: **{cba['max_years']} years**"
                )
                rule35_note = "35+ rule applies — cap recapture risk if player retires early." if cba["hits_35_rule"] else "No 35+ rule concerns."
                rec_col2.success(
                    f"Model recommendation: **{cba['recommended']} years**  \n"
                    f"Based on age {curr_age:.0f} trajectory.  \n"
                    f"{rule35_note}"
                )

                # Download
                csv = proj_df.to_csv(index=False)
                st.download_button(
                    "Download contract projection CSV", data=csv,
                    file_name=f"{pred['matched'].replace(' ','_')}_{contract_team}_{n_years}yr_contract.csv",
                    mime="text/csv"
                )

    # ── Model Info ────────────────────────────────────────────────────────────────
    with tab_model:
        st.markdown("#### Offensive Model Cache")
        if st.button("Retrain offensive model (deletes cache)"):
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            for key in ["df","team_ctx","has_age","player_profiles","fit_models",
                        "fit_metrics","fit_feature_names","next_models","next_metrics","next_feature_names"]:
                st.session_state.pop(key, None)
            st.success("Offensive cache cleared — refresh the page to retrain.")

        st.divider()
        with st.expander("Team Fit model quality", expanded=True):
            show_metrics(fit_metrics, "Team Fit")
        with st.expander("Next Season model quality"):
            show_metrics(next_metrics, "Next Season")
        with st.expander("Team Fit — feature importance"):
            st.pyplot(make_importance_chart(fit_models, fit_feature_names))
        with st.expander("Next Season — feature importance"):
            st.pyplot(make_importance_chart(next_models, next_feature_names))

        if def_models_loaded:
            st.divider()
            st.markdown("#### Defensive Model Cache")
            if st.button("Retrain defensive models"):
                if os.path.exists(DEF_CACHE_FILE):
                    os.remove(DEF_CACHE_FILE)
                for k in list(st.session_state.keys()):
                    if k.startswith("def_") or k.startswith("dpred_"):
                        del st.session_state[k]
                st.success("Defensive cache cleared — refresh to retrain.")
            with st.expander("Defensive Current Fit quality", expanded=True):
                def_show_metrics(def_fit_metrics, "Defensive Current Fit")
            with st.expander("Defensive Next Season quality"):
                def_show_metrics(def_next_metrics, "Defensive Next Season")

    # ── Validation (nested) ───────────────────────────────────────────────────────
    with tab_val:
        val_t1, val_t2 = st.tabs(["Offensive", "Defensive"])

        with val_t1:
            st.subheader("2025-26 Offensive Validation")
            st.caption(
                "Pulls live 2025-26 stats from the NHL API and compares against "
                "the model predictions. Points and Goals converted to per-game rates."
            )
            if st.button("Refresh NHL API stats"):
                st.cache_data.clear()
                for k in ["edge_weighted_shots"]:
                    st.session_state.pop(k, None)
                st.rerun()

            actual_df, err = fetch_nhl_current_season()
            if err:
                st.error(f"Could not fetch NHL API data: {err}")
            elif actual_df is not None:
                st.success(f"Fetched {len(actual_df):,} skaters with 10+ games played.")
                # weighted_shots_pg not used in validation — points and goals only

                with st.spinner("Comparing predictions to actual stats..."):
                    val_df = build_validation_results(
                        actual_df, df, team_ctx, fit_models, player_profiles, has_age
                    )
                if val_df.empty:
                    st.warning("No players matched between NHL API and model profiles.")
                else:
                    st.markdown(f"**{len(val_df):,} players matched**")
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                    fig.patch.set_facecolor("#0e1117")
                    make_scatter(val_df, "actual_points_gp", "pred_points_gp", "Points / Game",  axes[0])
                    make_scatter(val_df, "actual_goals_gp",  "pred_goals_gp",  "Goals / Game",   axes[1])
                    plt.tight_layout()
                    st.pyplot(fig)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Points/GP MAE",
                              f"{mean_absolute_error(val_df['actual_points_gp'], val_df['pred_points_gp']):.3f}")
                    c2.metric("Goals/GP MAE",
                              f"{mean_absolute_error(val_df['actual_goals_gp'], val_df['pred_goals_gp']):.3f}")
                    c3.metric("Players matched", f"{len(val_df):,}")

                    pred_range    = val_df["pred_points_gp"].max() - val_df["pred_points_gp"].min()
                    actual_range  = val_df["actual_points_gp"].max() - val_df["actual_points_gp"].min()
                    compression   = pred_range / actual_range if actual_range > 0 else np.nan
                    slope_val     = calibration_slope(val_df, "actual_points_gp", "pred_points_gp")
                    c3b, c4b = st.columns(2)
                    c3b.metric("Prediction Spread Ratio", f"{compression:.2%}" if not pd.isna(compression) else "n/a")
                    c4b.metric("Calibration Slope",       f"{slope_val:.2f}"   if not pd.isna(slope_val)   else "n/a")

                    elite_pts_mae, elite_pts_bias, elite_pts_n = elite_segment_stats(
                        val_df, "actual_points_gp", "pred_points_gp", quantile=ELITE_QUANTILE)
                    elite_goals_mae, elite_goals_bias, elite_goals_n = elite_segment_stats(
                        val_df, "actual_goals_gp", "pred_goals_gp", quantile=ELITE_QUANTILE)
                    e1, e2 = st.columns(2)
                    e1.metric(f"Elite Points/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                              f"{elite_pts_mae:.3f}" if not pd.isna(elite_pts_mae) else "n/a",
                              f"bias {elite_pts_bias:+.3f}" if not pd.isna(elite_pts_bias) else None)
                    e2.metric(f"Elite Goals/GP MAE (top {int((1-ELITE_QUANTILE)*100)}%)",
                              f"{elite_goals_mae:.3f}" if not pd.isna(elite_goals_mae) else "n/a",
                              f"bias {elite_goals_bias:+.3f}" if not pd.isna(elite_goals_bias) else None)
                    st.caption(f"Elite sample sizes: Points {elite_pts_n}, Goals {elite_goals_n}")
                    st.divider()
                    st.markdown("#### Biggest Misses")
                    misses = val_df.reindex(val_df["points_gp_error"].abs().nlargest(15).index)[
                        ["player_name","team","games_played","actual_points_gp",
                         "pred_points_gp","points_gp_error","actual_goals_gp","pred_goals_gp","seasons_used"]]
                    st.dataframe(misses, use_container_width=True)
                    st.markdown("#### Best Predictions")
                    best = val_df.reindex(val_df["points_gp_error"].abs().nsmallest(15).index)[
                        ["player_name","team","games_played","actual_points_gp",
                         "pred_points_gp","points_gp_error","actual_goals_gp","pred_goals_gp","seasons_used"]]
                    st.dataframe(best, use_container_width=True)
                    csv = val_df.to_csv(index=False)
                    st.download_button("Download full validation CSV", data=csv,
                                       file_name="validation_2025_26.csv", mime="text/csv")

        with val_t2:
            st.subheader("2025-26 Defensive Validation")
            st.caption(
                "Compares defensive model predictions against 2025-26 actual stats "
                "from the NHL API realtime endpoint (hits and takeaways per game)."
            )
            if st.button("Refresh defensive stats"):
                fetch_nhl_defensive_stats.cache_clear()

            if not def_models_loaded:
                st.warning("Defensive model not loaded.")
            else:
                def_actual, def_err = fetch_nhl_defensive_stats()
                if def_err:
                    st.error(f"Could not fetch NHL API data: {def_err}")
                elif def_actual is not None:
                    def_actual_d = def_actual[
                        def_actual["player_id"].isin(def_player_profiles.keys())
                    ].copy()
                    st.success(f"Fetched {len(def_actual_d):,} defensemen with 10+ games played.")
                    with st.spinner("Comparing predictions to actual stats..."):
                        def_val_df = build_defensive_validation(
                            def_actual_d, def_df, def_team_ctx,
                            def_fit_models, def_player_profiles, def_has_age,
                            feature_names=def_fit_feature_names
                        )
                    if def_val_df.empty:
                        st.warning("No defensemen matched between NHL API and model profiles.")
                    else:
                        st.markdown(f"**{len(def_val_df):,} defensemen matched**")

                        has_pim = "actual_pim_pg" in def_val_df.columns and def_val_df["actual_pim_pg"].sum() > 0
                        has_xga = "actual_xga_per60" in def_val_df.columns and def_val_df["actual_xga_per60"].sum() > 0

                        metric_cols = st.columns(4)
                        metric_cols[0].metric("Hits/GP MAE",
                                  f"{mean_absolute_error(def_val_df['actual_hits_pg'], def_val_df['pred_hits_pg']):.3f}")
                        metric_cols[1].metric("Takeaways/GP MAE",
                                  f"{mean_absolute_error(def_val_df['actual_tk_pg'], def_val_df['pred_tk_pg']):.3f}")
                        if has_pim:
                            metric_cols[2].metric("PIM/GP MAE",
                                      f"{mean_absolute_error(def_val_df['actual_pim_pg'], def_val_df['pred_pim_pg']):.3f}")
                        metric_cols[3].metric("Defensemen matched", f"{len(def_val_df):,}")
                        st.caption("PIM/GP: actual = penaltyMinutes/GP from NHL API, predicted = ind_penalty_minutes_pg from MoneyPuck.")

                        # 2x2 grid — hits, takeaways, xGA, PIM
                        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                        fig.patch.set_facecolor("#0e1117")
                        make_scatter(def_val_df, "actual_hits_pg", "pred_hits_pg", "Hits / Game",      axes[0][0])
                        make_scatter(def_val_df, "actual_tk_pg",   "pred_tk_pg",   "Takeaways / Game", axes[0][1])
                        if has_pim:
                            make_scatter(def_val_df, "actual_pim_pg", "pred_pim_pg", "PIM / Game", axes[1][0])
                        else:
                            axes[1][0].set_facecolor("#0e1117")
                            axes[1][0].text(0.5, 0.5, "PIM data not available\nfrom NHL API",
                                            ha="center", va="center", color="white", fontsize=12,
                                            transform=axes[1][0].transAxes)
                            axes[1][0].set_title("PIM / Game", color="white")
                        axes[1][1].set_facecolor("#0e1117")
                        axes[1][1].text(0.5, 0.5, "xGA validation requires\nMoneyPuck current season data",
                                        ha="center", va="center", color="white", fontsize=12,
                                        transform=axes[1][1].transAxes)
                        axes[1][1].set_title("xGA Against / 60", color="white")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        st.divider()
                        st.markdown("#### Biggest Misses (Hits/GP)")
                        miss_cols = ["player_name","team","games_played",
                                     "actual_hits_pg","pred_hits_pg","hits_error",
                                     "actual_tk_pg","pred_tk_pg","tk_error",
                                     "actual_pim_pg","pred_pim_pg","seasons_used"]
                        misses = def_val_df.reindex(def_val_df["hits_error"].abs().nlargest(15).index)[
                            [c for c in miss_cols if c in def_val_df.columns]]
                        st.dataframe(misses, use_container_width=True)

                        st.markdown("#### Best Predictions (Hits/GP)")
                        best = def_val_df.reindex(def_val_df["hits_error"].abs().nsmallest(15).index)[
                            [c for c in miss_cols if c in def_val_df.columns]]
                        st.dataframe(best, use_container_width=True)

                        csv = def_val_df.to_csv(index=False)
                        st.download_button("Download defensive validation CSV", data=csv,
                                           file_name="def_validation_2025_26.csv", mime="text/csv")

# ── Introduction ─────────────────────────────────────────────────────────────
if active_tab == "Introduction":
    st.subheader("Introduction")
    st.caption("An abstract and overview of the project.")
    st.markdown(
        """
        For this project I wanted to build something that could actually help answer a question that I think a lot of hockey fans and even front offices are not fully sure how to approach, which is how accurately can historical NHL data predict offensive performance for forwards and defensive impact for defensemen and how can those predictions be used to project how a player might perform if they were to change teams.
        
        The reason I wanted to focus on this specifically is because I think there is a lot of uncertainty around how a player is going to perform when they change teams and while the numbers can tell us a lot they do not always give us the full picture, so I wanted to see if building a model around historical data could help close that gap.
        
        The tool is designed to be used by NHL GMs, fantasy hockey players, and anyone who is into hockey analytics because I wanted it to be something that felt useful across different types of people who care about the game in different ways. It uses a machine learning model trained on historical NHL data to predict performance for any player so that when a trade or signing happens the data is already there to project how they are going to do in a new context. It also includes a contract evaluator that takes those projections and uses age curves to assess whether a player's performance is actually worth what they are being paid, which I thought was important to add because predicting performance only tells part of the story if you are a GM or a fantasy player trying to make a decision. On top of that there is a player search interface and a validation section that compares the model's predictions against live NHL API data from the current season so you can see how well it is actually working.
        ---

        **Project:** NHL Player Predictor
        How accurately can historical NHL data predict offensive performance for forwards and defensive impact for defensemen, and how can those predictions be used to project how a player might perform if they were to change teams?


        """
    )

# ── Literature Review ─────────────────────────────────────────────────────────
if active_tab == "Literature Review":
    st.subheader("Literature Review")
    st.caption("An overview of existing research and sources relevant to this project.")
    st.markdown(
        """
        There is actually a decent amount of work out there that touches on different pieces of what this project is trying to do, even if nothing does all of it together in one place.
        
        The most directly relevant starting point is the NHL EDGE stats breakdown of Mikko Rantanen's outlook after his trade to the Hurricanes published on NHL.com. What I liked about this piece is that it goes pretty deep on how Rantanen performed with Colorado and what his underlying numbers looked like before the trade. The abeck2309 NHL trade ROI project connects to this well because it takes that same idea and builds a model around it, using realized and expected goals above replacement to put an actual number on whether a trade worked out. The problem with both of them though is they are looking backwards at what already happened and neither one is projecting how a player is going to do going forward in a new system with new linemates, which is exactly the gap this project is trying to close.
        
        On the data side, the Hockey Analytics article on pulling data directly from the NHL API was really useful because it walks through how to actually get the data without having to worry about a lot of the finer details of scraping. This connects to something like the AAZZAZRON TradeTracker project which is a Discord bot that scrapes Sportsnet to pull in recent trade and signing details automatically. That one is a little out of scope for where the project is right now but is worth keeping in mind for future versions when financial data becomes more important.
        
        The most important thing to note is that all of this data needs to be validated and cleaned before it can be used in the model. This is where the NHL API comes in, as it provides a reliable source of real-time data that can be used to verify the accuracy of the historical data.
        
        Beyond those specific sources there is a broader body of hockey analytics research that is relevant here. Work on expected goals models and wins above replacement in hockey has shown that raw counting stats like goals and assists do not always tell the full story of how valuable a player actually is, which is part of why this project focuses on underlying production metrics rather than just point totals. There is also research on how team context affects individual performance, specifically how linemates, zone deployment, and coaching systems can inflate or suppress a player's numbers in ways that make it hard to project them into a new situation. This is the core challenge the model is trying to account for. Finally there is existing work on age curves in hockey which shows that forwards typically peak in their mid-twenties and decline gradually after that, which is the foundation for how the contract evaluator in this project works.
        """
    )

# ── Methodology ───────────────────────────────────────────────────────────────
if active_tab == "Methodology":
    st.subheader("Research Methodology")
    st.caption("A walkthrough of the specific techniques and methods used in this project.")
    
    st.markdown("""
    *This section describes the specific techniques and methods used, connecting methodology
    directly to the Research Question. It includes methods of data collection, analysis, and
    the choices made to refine or limit the project.*
    """)

    st.markdown("---")
    st.markdown("#### Data Collection")
    st.markdown("""
    The data for this project comes from moneypuck.com which has fully downloadable historical NHL data going back to the 2008-2009 season all the way up to 2024-2025, and I validated it using the NHL API to make sure what I was working with was accurate. Moneypuck also provides lines data which I used to better understand the context of each player's performance, and it does have up to date stats available as well, but I chose not to include those yet because I wanted to see how the model performed on historical data first and then use the API to validate those predictions before adding anything else.

    I split forwards and defensemen into separate datasets because they have very different roles and I wanted to look for different primary stats for each, so keeping them together would have made the model less accurate and would have added a lot of stats that were not relevant to what I was actually trying to predict. From there I pulled out the offensive stats that I felt had the most impact on points and goals per game and fed those into the model, cutting out anything that was not essential so the datasets stayed manageable. I also filtered out players who did not reach a minimum number of games or minutes played because I wanted to make sure I was only working with players who actually had a real sample size and not guys who only played a handful of games.
    """)

    st.markdown("#### Data Management")
    st.markdown("""
    One thing I had to figure out early on was what to do with missing values in the data. Rather than just dropping those rows entirely I converted any NAs to zero because since all the features are numerical a missing stat is basically the same as zero production in that category, and deleting the whole row would have thrown away a lot of valid data that was still useful. I also joined age data to the main dataset using player ID and season as the merge keys rather than player name because names can have spelling variations and special characters especially for international players, so using the ID just made it cleaner and avoided mismatches. To keep the app running smoothly I also saved the trained models as joblib files so they load directly at runtime instead of retraining every single time someone opens the page.
    """)

    st.markdown("---")

    # ── Technical breakdown PDF download ─────────────────────────────────────
    _pdf_path = os.path.join(os.path.dirname(__file__), "analysis_technical.pdf")
    if os.path.exists(_pdf_path):
        with open(_pdf_path, "rb") as _pdf_file:
            st.download_button(
                label="📄 Technical Breakdown — download full model documentation (PDF)",
                data=_pdf_file,
                file_name="analysis_technical.pdf",
                mime="application/pdf",
                help="In-depth technical documentation: LightGBM details, feature weights, "
                     "residual modeling, cross-validation setup, and training parameters.",
            )

    st.markdown("#### Analysis & Modeling")
    st.markdown("""
                This section explains how the prediction model works and you do not need a technical background to follow along, and if you do want the more technical side of things you can download the PDF above.
                """)
    st.markdown("**What does it do?**")
    st.markdown("""
                You put in a player's history and skill profile and the model estimates how many points, goals, and overall contributions they are likely to put up, both in general and specifically on any of the 32 NHL teams. I built separate models for forwards and defensemen because they have very different roles and I wanted to make sure the model was actually looking for the right things for each.
                """)
    st.markdown("**What is it actually predicting?**")
    st.markdown("""
                Rather than just predicting a raw stat line the model is trying to figure out something more useful, which is whether a player is going to outperform or underperform their own historical baseline. Every player gets a personal benchmark built from their career history and recent seasons and the model's job is to figure out whether their skills and team fit are going to push them above or below that mark. I thought this was a better approach than just predicting that good players score more which is not really telling you anything useful.
                """)
    st.markdown("**What information goes in?**")
    st.markdown("""
                The model pulls from four main areas. First is shooting and skill, so how dangerous is this player's shot, how often are they beating goalies relative to what you would expect, and how much are they contributing on the power play. Second is career history, so what have they done across their career, what did last season look like, and whether they are trending up or down. Third is age and career stage because a 25 year old and a 33 year old with the same stats are in very different situations and the model accounts for that. Fourth is team and system fit because ice time, line quality, and shot generation all vary a lot by organization and the model can swap in any of the 32 teams to simulate how a player would do in a different system, which is really the core of what makes this useful for trades and signings.
                """)
    st.markdown("**Two model versions**")
    st.markdown("""
                The Team Fit model is for right now. You give it a player's current skill profile and it tells you how they would produce on each team today, so it is best for trade deadline and free agency decisions. The Next Season model is for planning ahead because it adds trajectory signals like year over year stat changes to figure out whether a player is still improving or starting to decline, which makes it better for long term contracts and draft planning.
                """)

    st.markdown("**How does it work?**")
    st.markdown("""
                The model uses machine learning algorithms to analyze the input data and identify patterns and relationships. It then applies these patterns to make predictions about future performance.
                """)
    st.markdown("**How is it trained and validated?**")
    st.markdown("""
                The model gets tested by training on some historical seasons and predicting others, rotating three times so every data point gets a chance to be evaluated. I also made a few deliberate choices to make sure it is most accurate where it actually matters, so players with very few games are left out to avoid noisy samples and elite players are weighted more heavily during training so the model gets sharper at the predictions that actually affect roster decisions the most.
                """)
    st.markdown("---")
    st.markdown("#### Choices & Limitations")
    st.markdown("""Goalies and low-minutes players were excluded from the model because they don't
    provide a reliable sample for predictions — goalies in particular use entirely
    different performance metrics. A broader limitation is the inherent unpredictability
    of the NHL itself. Players and teams are constantly evolving, coaches and systems
    change, and those shifts can affect performance in ways the model can't fully
    anticipate. The model's accuracy is strong, but it's not perfect, and there will
    always be factors outside the data.

    Contract dollar values were also left out. Partly this came down to time — finding
    a clean, free salary datasource proved difficult — but more fundamentally, each
    team values players differently based on their own needs and circumstances. Without
    a reliable way to model that context, adding salary data risked making the
    contract recommendations less accurate rather than more.
    """)

    st.markdown("---")
    st.markdown("#### AI Tool Usage")
    st.markdown("""
    Claude was used throughout development to help with code generation, debugging,
    and model design. Its ability to produce working code quickly was especially valuable
    given the volume of Streamlit logic involved, and it was a useful guide for someone
    with limited prior Streamlit experience.

    That said, there were real limitations. Claude didn't always understand exactly
    what data was available or how it needed to be structured, which meant generated
    code often required manual verification and adjustment. There were also points where
    Claude's suggested approach to the model differed from what the data actually
    supported — in early testing the model had a performance cap that was suppressing
    true high-end predictions, and catching that required understanding the underlying
    data well enough to recognize the problem. AI assistance works best when you still
    know what the right answer should look like.
    """)
if active_tab == "Analysis & Findings":
    st.subheader("Analysis & Findings")
    st.caption("What was discovered through the analysis.")

    st.markdown("#### Key Findings")
    st.markdown("""
    Overall I think the model performed pretty well. Across 427 matched players the Points/GP MAE came in at 0.123 and the Goals/GP MAE came in at 0.074 which means on average the model is off by about an eighth of a point per game on points and less than a tenth of a goal per game on goals. For a model trying to predict individual player production across an entire season I think that is a solid result.

    The prediction spread ratio came in at 95% which means the model is generating predictions that cover a realistic range rather than just clustering everything in the middle, and the calibration slope of 0.90 means it is tracking pretty closely with how players actually performed with just a slight tendency to be conservative on the higher end.
    """)

    st.markdown("#### Elite Player Predictions")
    st.markdown("""
    The trickiest part of the model was predicting elite players and that shows up in the numbers. For the top 9% of players the Points/GP MAE goes up to 0.154 with a bias of -0.102, and the Goals/GP MAE goes up to 0.105 with a bias of -0.075. The negative bias means the model is consistently undershooting on elite players which makes sense because truly elite production is harder to predict from historical data alone. This is something I would want to improve in future versions, but given that elite players were already weighted three times more heavily during training I think this is about as good as you can get without adding more contextual data.
    """)

    st.markdown("#### Defensive Metrics")
    st.markdown("""
    Looking at the scatter plots the model does best on Hits/Game with an MAE of 0.258 and a correlation of 0.89 which is really strong and shows the model is picking up on physical play patterns well. Takeaways/Game and PIM/Game were harder to predict with correlations of 0.60 and 0.58 respectively, which honestly makes sense because those stats are a lot more random and situational than something like hits. A player can take a penalty or win a puck battle in ways that are hard to predict from historical trends alone so I was not too surprised to see the model struggle a bit more there.
    """)

    st.markdown("#### Surprises & Anomalies")
    st.markdown("""
    The biggest surprise was how well the model handled the spread of predictions. I was worried early on that it was going to cluster everything around the mean and not really differentiate between players, which is a common problem with this type of model. The 95% prediction spread ratio shows that it is not doing that which I think is the most important thing to get right if you actually want to use this for roster decisions. The elite underprediction is the main thing I would flag as something to keep an eye on because if you are a GM trying to evaluate a superstar trade that bias could matter.
    """)
# ── Conclusion ────────────────────────────────────────────────────────────────
if active_tab == "Conclusion":
    st.subheader("Conclusion")
    st.caption("A summary of what was built and what it means.")

    st.markdown("""
    Going back to the original research question of how accurately can historical NHL data predict player performance and how can those predictions be used to project how a player might perform if they change teams, I think the honest answer is that this project does not answer that question directly but instead builds something that lets anyone answer it themselves.
    """)

    st.markdown("""
    What I built is a tool that takes a player's historical data and gives you a projection of how they would perform on any of the 32 NHL teams right now or going into next season. On top of that the contract evaluator takes those performance projections and uses the player's age curve to project how long a contract should be, which I think is one of the most practically useful parts of the whole app because knowing how a player will perform is only half the decision. The other half is figuring out how long you can count on that level of production, and that is what the contract side is trying to help with.
    """)

    st.markdown("""
    So rather than me running a single study on one trade and calling the question answered, anyone using this app can plug in any player and get that answer for whatever situation they are actually looking at. I think that is more useful in practice because the question is going to look different depending on whether you are a GM at the trade deadline, a fantasy player making a pickup, or just someone curious about what would happen if a player switched teams.
    """)

    st.markdown("""
    The model's validation results show that it is accurate enough to be genuinely useful and the elite player underprediction is something I would want to keep working on. But the foundation is there and the goal from the start was to build something that could be a real tool for real decisions, and I think that is what it is.
    """)

# ── Works Cited ───────────────────────────────────────────────────────────────
if active_tab == "Works Cited":
    st.subheader("Works Cited")
    st.caption("Sources used throughout this project.")

    # MLA 9th edition — one flat alphabetical list, hanging-indent style via HTML.
    _cite = (
        "<p style='margin:0 0 1.1em 0; padding-left:2em; text-indent:-2em; "
        "font-size:15px; line-height:1.6;'>{}</p>"
    )

    st.markdown(
        # 1. AAZZAZRON (AA)
        _cite.format(
            "AAZZAZRON. \"TradeTracker: A Discord Bot That Scrapes Sportsnet to Find "
            "the Most Recent NHL Trades and Signings.\" <em>GitHub</em>, "
            "<a href='https://github.com/AAZZAZRON/TradeTracker'>"
            "github.com/AAZZAZRON/TradeTracker</a>. Accessed 2025."
        ) +
        # 2. abeck2309 (AB)
        _cite.format(
            "abeck2309. \"nhl-trade-roi-xgar: Evaluating NHL Trades Using Realized "
            "and Expected xGAR.\" <em>GitHub</em>, "
            "<a href='https://github.com/abeck2309/nhl-trade-roi-xgar'>"
            "github.com/abeck2309/nhl-trade-roi-xgar</a>. Accessed 2025."
        ) +
        # 3. Anthropic (AN)
        _cite.format(
            "Anthropic. <em>Claude</em>, version claude-sonnet-4-20250514, "
            "Anthropic, 2025, "
            "<a href='https://claude.ai'>claude.ai</a>. Accessed 2025."
        ) +
        # 4. "Hockey Analytics..." (H)
        _cite.format(
            "\"Hockey Analytics &ndash; Getting Data Directly from the NHL API.\" "
            "<em>Hockey-Statistics</em>, 14 May 2025, "
            "<a href='https://hockey-statistics.com/2025/05/14/hockey-analytics-getting-data-directly-from-the-nhl-api/'>"
            "hockey-statistics.com/2025/05/14/hockey-analytics-getting-data-directly-from-the-nhl-api/</a>."
        ) +
        # 5. MoneyPuck (M)
        _cite.format(
            "MoneyPuck. \"MoneyPuck.com.\" <em>MoneyPuck</em>, "
            "<a href='https://moneypuck.com'>moneypuck.com</a>. Accessed 2025."
        ) +
        # 6. National Hockey League (NA)
        _cite.format(
            "National Hockey League. \"NHL Stats API.\" <em>NHL</em>, "
            "<a href='https://api-web.nhle.com'>api-web.nhle.com</a>. Accessed 2025."
        ) +
        # 7. "NHL EDGE Stats..." (NH)
        _cite.format(
            "\"NHL EDGE Stats: Rantanen's Outlook after Trade to Hurricanes.\" "
            "<em>NHL</em>, "
            "<a href='https://www.nhl.com/news/edge-stats-impact-of-trade-on-mikko-rantanen-martin-necas'>"
            "www.nhl.com/news/edge-stats-impact-of-trade-on-mikko-rantanen-martin-necas</a>. Accessed 2025."
        ),
        unsafe_allow_html=True,
    )