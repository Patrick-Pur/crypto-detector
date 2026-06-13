import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

ANALYSIS_FILE = "btc_analysis.csv"
SIGNALS_FILE = "bullish_early_signals.csv"

HORIZONS = [30, 90, 180]


# ============================================================
# 2. SEITENKONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BTC Bullish Reversal Dashboard",
    page_icon="₿",
    layout="wide"
)

st.title("₿ BTC Bullish Reversal Dashboard")

st.caption(
    "Research-Dashboard für bullische Frühwarnsignale. "
    "Kein automatisches Kauf- oder Verkaufssignal."
)


# ============================================================
# 3. DATEN LADEN
# ============================================================

@st.cache_data
def load_data():
    if not os.path.exists(ANALYSIS_FILE):
        raise FileNotFoundError(
            f"{ANALYSIS_FILE} fehlt. "
            "Bitte zuerst 'python main.py' ausführen."
        )

    if not os.path.exists(SIGNALS_FILE):
        raise FileNotFoundError(
            f"{SIGNALS_FILE} fehlt. "
            "Bitte zuerst 'python main.py' ausführen."
        )

    analysis = pd.read_csv(
        ANALYSIS_FILE,
        parse_dates=["Date"],
        index_col="Date"
    )

    signals = pd.read_csv(
        SIGNALS_FILE,
        parse_dates=["DATUM"]
    )

    return analysis, signals


try:
    analysis, signals = load_data()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


# ============================================================
# 4. HILFSFUNKTIONEN
# ============================================================

def to_bool(value):
    if isinstance(value, str):
        return value.lower() == "true"

    return bool(value)


def calculate_score(row):
    price = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])

    rsi = float(row["RSI"])

    macd = float(row["MACD"])
    macd_signal = float(row["MACD_SIGNAL"])

    drawdown = float(row["DRAWDOWN"])
    vol_ratio = float(row["VOL_RATIO"])

    score = 50

    if price > ema20 and ema20 > ema50:
        score += 20

    elif price < ema20 and ema20 < ema50:
        score -= 20

    if price > ema200:
        score += 10

    else:
        score -= 10

    if drawdown < -60:
        score += 25

    elif drawdown < -45:
        score += 15

    elif drawdown < -30:
        score += 5

    if rsi < 30:
        score += 15

    elif rsi > 70:
        score -= 15

    elif rsi > 50:
        score += 5

    elif rsi < 50:
        score -= 5

    if macd > macd_signal:
        score += 15

    else:
        score -= 15

    if vol_ratio >= 1.3:
        score += 10

    return max(
        0,
        min(100, score)
    )


def create_summary_table(signals_data):
    records = []

    for horizon in HORIZONS:
        column = f"RET_{horizon}D"

        if column not in signals_data.columns:
            continue

        values = pd.to_numeric(
            signals_data[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        wins = values[
            values > 0
        ]

        losses = values[
            values <= 0
        ]

        if not losses.empty:
            profit_factor = (
                wins.sum() /
                abs(losses.sum())
            )

        else:
            profit_factor = np.nan

        records.append(
            {
                "Horizont": f"{horizon} Tage",
                "Ereignisse": len(values),
                "Trefferquote": (
                    len(wins) /
                    len(values) *
                    100
                ),
                "Ø Rendite": values.mean(),
                "Median": values.median(),
                "Beste Rendite": values.max(),
                "Schlechteste Rendite": values.min(),
                "Profit Factor": profit_factor
            }
        )

    return pd.DataFrame(records)


def create_price_chart(
    chart_data,
    signal_data,
    log_scale
):
    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="BTC"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["EMA20"],
            mode="lines",
            name="EMA20"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["EMA50"],
            mode="lines",
            name="EMA50"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["EMA200"],
            mode="lines",
            name="EMA200"
        )
    )

    if not signal_data.empty:
        figure.add_trace(
            go.Scatter(
                x=signal_data["DATUM"],
                y=signal_data["BTC_PREIS"],
                mode="markers",
                name="EARLY-Signal",
                marker={
                    "symbol": "triangle-up",
                    "size": 14
                }
            )
        )

    figure.update_layout(
        title="BTC-Kurs mit EMA-Linien und EARLY-Signalen",
        height=650,
        xaxis_rangeslider_visible=False,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    if log_scale:
        figure.update_yaxes(
            type="log"
        )

    return figure


def create_rsi_chart(chart_data):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["RSI"],
            mode="lines",
            name="RSI"
        )
    )

    figure.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Überverkauft: RSI 30"
    )

    figure.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Überkauft: RSI 70"
    )

    figure.update_layout(
        title="RSI",
        height=350,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    figure.update_yaxes(
        range=[0, 100]
    )

    return figure


def create_drawdown_chart(chart_data):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["DRAWDOWN"],
            mode="lines",
            fill="tozeroy",
            name="Drawdown"
        )
    )

    figure.add_hline(
        y=-40,
        line_dash="dash",
        annotation_text="WATCH-Grenze: -40 %"
    )

    figure.update_layout(
        title="Drawdown vom bisherigen All-Time-High",
        height=350,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    return figure


# ============================================================
# 5. AKTUELLER STATUS
# ============================================================

latest = analysis.iloc[-1]

latest_date = analysis.index[-1].date()

price = float(latest["Close"])
rsi = float(latest["RSI"])
drawdown = float(latest["DRAWDOWN"])
vol_ratio = float(latest["VOL_RATIO"])

score = calculate_score(
    latest
)

watch_signal = to_bool(
    latest["WATCH_SIGNAL"]
)

early_signal = to_bool(
    latest["EARLY_SIGNAL"]
)


st.subheader("Aktueller Marktstatus")

st.caption(
    f"Letzte vollständig ausgewertete Tageskerze: "
    f"{latest_date}"
)

column_1, column_2, column_3, column_4, column_5 = (
    st.columns(5)
)

column_1.metric(
    "BTC-Kurs",
    f"{price:,.0f} USD"
)

column_2.metric(
    "RSI",
    f"{rsi:.1f}"
)

column_3.metric(
    "Drawdown",
    f"{drawdown:.1f} %"
)

column_4.metric(
    "Volumen-Faktor",
    f"{vol_ratio:.2f}x"
)

column_5.metric(
    "Umkehr-Score",
    f"{score}/100"
)


if early_signal:
    st.success(
        "EARLY: Bullische Frühbestätigung aktiv. "
        "WATCH-Kriterien erfüllt und MACD liegt über der Signallinie."
    )

elif watch_signal:
    st.warning(
        "WATCH: Markt ist überverkauft und mindestens 40 % "
        "unter dem bisherigen Hoch. Die bullische Bestätigung fehlt noch."
    )

else:
    st.info(
        "Kein qualifiziertes bullisches Frühwarnsignal aktiv."
    )


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.header(
    "Darstellung"
)

lookback_option = st.sidebar.selectbox(
    "Angezeigter Zeitraum",
    [
        "180 Tage",
        "365 Tage",
        "730 Tage",
        "Gesamter Zeitraum"
    ],
    index=1
)

log_scale = st.sidebar.checkbox(
    "Logarithmische Preisskala",
    value=False
)

if lookback_option == "180 Tage":
    chart_data = analysis.tail(
        180
    )

elif lookback_option == "365 Tage":
    chart_data = analysis.tail(
        365
    )

elif lookback_option == "730 Tage":
    chart_data = analysis.tail(
        730
    )

else:
    chart_data = analysis.copy()


signals_visible = signals[
    signals["DATUM"] >=
    chart_data.index.min()
].copy()


# ============================================================
# 7. TABS
# ============================================================

tab_overview, tab_signals, tab_backtest = st.tabs(
    [
        "Charts",
        "Historische EARLY-Signale",
        "Backtest"
    ]
)


with tab_overview:
    st.plotly_chart(
        create_price_chart(
            chart_data,
            signals_visible,
            log_scale
        ),
        use_container_width=True
    )

    st.plotly_chart(
        create_rsi_chart(
            chart_data
        ),
        use_container_width=True
    )

    st.plotly_chart(
        create_drawdown_chart(
            chart_data
        ),
        use_container_width=True
    )


with tab_signals:
    st.subheader(
        "Historische EARLY-Signalphasen"
    )

    display_columns = [
        "DATUM",
        "BTC_PREIS",
        "DRAWDOWN",
        "RSI",
        "VOL_RATIO",
        "VOL_LABEL",
        "RET_30D",
        "RET_90D",
        "RET_180D"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in signals.columns
    ]

    st.dataframe(
        signals[
            available_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="EARLY-Signale als CSV herunterladen",
        data=signals.to_csv(
            index=False
        ),
        file_name="bullish_early_signals.csv",
        mime="text/csv"
    )


with tab_backtest:
    st.subheader(
        "Zusammenfassung des EARLY-Backtests"
    )

    summary = create_summary_table(
        signals
    )

    formatted_summary = summary.copy()

    percentage_columns = [
        "Trefferquote",
        "Ø Rendite",
        "Median",
        "Beste Rendite",
        "Schlechteste Rendite"
    ]

    for column in percentage_columns:
        if column in formatted_summary.columns:
            formatted_summary[column] = (
                formatted_summary[column]
                .map(
                    lambda value:
                    f"{value:.1f} %"
                )
            )

    if "Profit Factor" in formatted_summary.columns:
        formatted_summary["Profit Factor"] = (
            formatted_summary["Profit Factor"]
            .map(
                lambda value:
                "-"
                if pd.isna(value)
                else f"{value:.2f}"
            )
        )

    st.dataframe(
        formatted_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Die Stichprobe ist bislang klein. "
        "Die Ergebnisse sind als Research-Auswertung zu lesen, "
        "nicht als belastbarer Nachweis einer profitablen Strategie."
    )