from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# 1. KONFIGURATION
# ============================================================

DATA_START = "2015-01-01"
ANALYSIS_START = "2017-01-01"

# Eng aufeinanderfolgende Signale werden zu einer Phase gebündelt.
COOLDOWN_DAYS = 30

# Neue Daten werden maximal alle 15 Minuten geladen.
CACHE_TTL_SECONDS = 900

HORIZONS = [30, 90, 180]

# Fallback, falls Yahoo Finance zeitweise keine Daten liefert.
FALLBACK_FILE = Path("btc_analysis.csv")


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
    "Keine Anlageempfehlung und kein automatisches Handelssignal."
)


# ============================================================
# 3. DATEN LADEN UND AUFBEREITEN
# ============================================================

def normalize_columns(data):
    """
    Vereinheitlicht die Spaltenstruktur und den Zeitindex.
    yfinance liefert teilweise MultiIndex-Spalten.
    """

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    data.index = pd.to_datetime(
        data.index
    )

    if getattr(
        data.index,
        "tz",
        None
    ) is not None:
        data.index = data.index.tz_localize(
            None
        )

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ]

    return data.sort_index()


def load_fallback_data():
    """
    Lädt den letzten gespeicherten CSV-Stand,
    falls der aktuelle Online-Abruf fehlschlägt.
    """

    if not FALLBACK_FILE.exists():
        return None

    fallback = pd.read_csv(
        FALLBACK_FILE,
        parse_dates=["Date"],
        index_col="Date"
    )

    return normalize_columns(
        fallback
    )


@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    show_spinner=False
)
def load_market_data():
    """
    Lädt BTC-Tagesdaten automatisch.
    Bei einem Fehler wird die CSV-Fallback-Datei verwendet.
    """

    source = "Yahoo Finance via yfinance"
    fallback_reason = None

    try:
        market_data = yf.download(
            "BTC-USD",
            start=DATA_START,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        market_data = normalize_columns(
            market_data
        )

        if market_data.empty:
            raise RuntimeError(
                "Yahoo Finance hat keine BTC-Daten geliefert."
            )

    except Exception as error:
        market_data = load_fallback_data()

        if market_data is None:
            raise RuntimeError(
                "Die aktuellen BTC-Daten konnten nicht geladen werden "
                "und es ist keine lokale Fallback-Datei vorhanden."
            ) from error

        source = "Fallback-Datei btc_analysis.csv"
        fallback_reason = str(error)

    required_price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_price_columns
        if column not in market_data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Folgende Kursdaten-Spalten fehlen: "
            + ", ".join(
                missing_columns
            )
        )

    market_data = market_data[
        required_price_columns
    ].copy()

    return (
        market_data,
        source,
        fallback_reason
    )


def add_indicators(raw_data):
    """
    Berechnet technische Indikatoren, Signale und Scores.
    """

    data = raw_data.copy()

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    data["EMA20"] = data["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    data["EMA50"] = data["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    data["EMA200"] = data["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_BULL_CROSS"] = (
        (
            data["MACD"] >
            data["MACD_SIGNAL"]
        )
        &
        (
            data["MACD"].shift(1) <=
            data["MACD_SIGNAL"].shift(1)
        )
    )

    data["MACD_BULL_CROSS_RECENT_7D"] = (
        data["MACD_BULL_CROSS"]
        .rolling(
            7,
            min_periods=1
        )
        .max()
        .astype(bool)
    )

    # --------------------------------------------------------
    # Volumen
    # --------------------------------------------------------

    data["VOL_MA20"] = data["Volume"].rolling(
        20
    ).mean()

    data["VOL_RATIO"] = (
        data["Volume"] /
        data["VOL_MA20"]
    )

    # --------------------------------------------------------
    # Drawdown vom bisherigen All-Time-High
    # --------------------------------------------------------

    data["ATH"] = data["Close"].cummax()

    data["DRAWDOWN"] = (
        data["Close"] /
        data["ATH"] - 1
    ) * 100

    # --------------------------------------------------------
    # Bullische Signalstufen
    # --------------------------------------------------------

    # WATCH:
    # BTC ist stark gefallen und kurzfristig überverkauft.
    data["WATCH_SIGNAL"] = (
        (
            data["RSI"] < 30
        )
        &
        (
            data["DRAWDOWN"] < -40
        )
    )

    # EARLY:
    # Zusätzlich beginnt das Momentum nach oben zu drehen.
    data["EARLY_SIGNAL"] = (
        data["WATCH_SIGNAL"]
        &
        (
            data["MACD"] >
            data["MACD_SIGNAL"]
        )
    )

    # --------------------------------------------------------
    # Setup-Score
    # --------------------------------------------------------
    #
    # Frage:
    # Wie deutlich ist die Stress- und Bodenbildungsphase?
    #
    # Maximum: 100 Punkte
    # --------------------------------------------------------

    drawdown_points = np.select(
        [
            data["DRAWDOWN"] <= -60,
            data["DRAWDOWN"] <= -50,
            data["DRAWDOWN"] <= -40,
            data["DRAWDOWN"] <= -30
        ],
        [
            45,
            35,
            25,
            10
        ],
        default=0
    )

    rsi_points = np.select(
        [
            data["RSI"] < 25,
            data["RSI"] < 30,
            data["RSI"] < 35,
            data["RSI"] < 45
        ],
        [
            35,
            30,
            20,
            10
        ],
        default=0
    )

    volume_points = np.select(
        [
            data["VOL_RATIO"] >= 1.5,
            data["VOL_RATIO"] >= 1.3,
            data["VOL_RATIO"] >= 1.0
        ],
        [
            20,
            15,
            8
        ],
        default=0
    )

    data["SETUP_SCORE"] = np.minimum(
        100,
        (
            drawdown_points +
            rsi_points +
            volume_points
        )
    ).astype(int)

    # --------------------------------------------------------
    # Bestätigungs-Score
    # --------------------------------------------------------
    #
    # Frage:
    # Drehen Momentum und kurzfristige Struktur bereits nach oben?
    #
    # Maximum: 100 Punkte
    # --------------------------------------------------------

    confirmation_score = np.zeros(
        len(data),
        dtype=int
    )

    confirmation_score += np.where(
        data["MACD"] >
        data["MACD_SIGNAL"],
        40,
        0
    )

    confirmation_score += np.where(
        data["MACD_BULL_CROSS_RECENT_7D"],
        20,
        0
    )

    confirmation_score += np.where(
        data["Close"] >
        data["EMA20"],
        15,
        0
    )

    confirmation_score += np.where(
        data["EMA20"] >
        data["EMA20"].shift(5),
        10,
        0
    )

    confirmation_score += np.where(
        data["RSI"] >
        data["RSI"].shift(3),
        10,
        0
    )

    confirmation_score += np.where(
        data["Close"] >
        data["Close"].shift(3),
        5,
        0
    )

    data["CONFIRMATION_SCORE"] = np.minimum(
        100,
        confirmation_score
    ).astype(int)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "VOL_MA20",
        "VOL_RATIO",
        "DRAWDOWN",
        "SETUP_SCORE",
        "CONFIRMATION_SCORE"
    ]

    data = data.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # Unvollständige aktuelle Tageskerze entfernen
    # --------------------------------------------------------

    today_utc = pd.Timestamp.now(
        tz="UTC"
    ).date()

    if (
        not data.empty
        and
        data.index[-1].date() >= today_utc
    ):
        data = data.iloc[:-1].copy()

    return data.loc[
        ANALYSIS_START:
    ].copy()


def select_signal_phases(
    data,
    signal_column,
    cooldown_days
):
    """
    Bündelt mehrere eng aufeinanderfolgende Signale
    zu unabhängigen Signalphasen.
    """

    signal_mask = data[
        signal_column
    ].fillna(
        False
    )

    phase_starts = (
        signal_mask
        &
        ~signal_mask.shift(
            1,
            fill_value=False
        )
    )

    candidate_dates = list(
        data.index[
            phase_starts
        ]
    )

    selected_dates = []

    for date in candidate_dates:
        if not selected_dates:
            selected_dates.append(
                date
            )

            continue

        distance = (
            date -
            selected_dates[-1]
        ).days

        if distance >= cooldown_days:
            selected_dates.append(
                date
            )

    return data.loc[
        selected_dates
    ].copy()


def calculate_event_table(
    data,
    signal_phases
):
    """
    Berechnet die Renditen nach historischen EARLY-Signalen.
    """

    records = []

    for date, row in signal_phases.iterrows():
        position = data.index.get_loc(
            date
        )

        entry_price = float(
            data.iloc[position][
                "Close"
            ]
        )

        record = {
            "DATUM": date,
            "BTC_PREIS": entry_price,
            "DRAWDOWN": float(
                row["DRAWDOWN"]
            ),
            "RSI": float(
                row["RSI"]
            ),
            "VOL_RATIO": float(
                row["VOL_RATIO"]
            ),
            "SETUP_SCORE": int(
                row["SETUP_SCORE"]
            ),
            "CONFIRMATION_SCORE": int(
                row["CONFIRMATION_SCORE"]
            )
        }

        for horizon in HORIZONS:
            if (
                position + horizon >=
                len(data)
            ):
                record[
                    f"RET_{horizon}D"
                ] = None

                record[
                    f"MAE_{horizon}D"
                ] = None

                record[
                    f"MFE_{horizon}D"
                ] = None

                continue

            future_price = float(
                data.iloc[
                    position + horizon
                ]["Close"]
            )

            window = data.iloc[
                position:
                position + horizon + 1
            ]["Close"]

            record[
                f"RET_{horizon}D"
            ] = (
                future_price /
                entry_price - 1
            ) * 100

            record[
                f"MAE_{horizon}D"
            ] = (
                float(
                    window.min()
                ) /
                entry_price - 1
            ) * 100

            record[
                f"MFE_{horizon}D"
            ] = (
                float(
                    window.max()
                ) /
                entry_price - 1
            ) * 100

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# 4. DATEN ABRUFEN
# ============================================================

with st.spinner(
    "BTC-Daten werden geladen und analysiert ..."
):
    try:
        raw_data, data_source, fallback_reason = load_market_data()

        analysis = add_indicators(
            raw_data
        )

    except Exception as error:
        st.error(
            f"Daten konnten nicht geladen werden: {error}"
        )

        st.stop()

if analysis.empty:
    st.error(
        "Nach der Berechnung stehen keine auswertbaren Daten zur Verfügung."
    )

    st.stop()

watch_phases = select_signal_phases(
    analysis,
    "WATCH_SIGNAL",
    COOLDOWN_DAYS
)

early_phases = select_signal_phases(
    analysis,
    "EARLY_SIGNAL",
    COOLDOWN_DAYS
)

early_events = calculate_event_table(
    analysis,
    early_phases
)


# ============================================================
# 5. HILFSFUNKTIONEN FÜR DIE OBERFLÄCHE
# ============================================================

def format_check(
    condition
):
    return (
        "✅ Erfüllt"
        if bool(condition)
        else "❌ Fehlt"
    )


def format_percent(
    value
):
    if pd.isna(
        value
    ):
        return "-"

    return f"{value:.1f} %"


def create_summary_table(
    events
):
    records = []

    for horizon in HORIZONS:
        column = f"RET_{horizon}D"

        if (
            events.empty
            or
            column not in events.columns
        ):
            continue

        values = pd.to_numeric(
            events[column],
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

        profit_factor = np.nan

        if (
            not wins.empty
            and
            not losses.empty
        ):
            profit_factor = (
                wins.sum() /
                abs(
                    losses.sum()
                )
            )

        records.append(
            {
                "Horizont": f"{horizon} Tage",
                "Ereignisse": len(
                    values
                ),
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

    return pd.DataFrame(
        records
    )


# ============================================================
# 6. CHARTS
# ============================================================

def create_price_chart(
    chart_data,
    visible_watch_phases,
    visible_early_phases,
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

    for column, label in [
        (
            "EMA20",
            "EMA20"
        ),
        (
            "EMA50",
            "EMA50"
        ),
        (
            "EMA200",
            "EMA200"
        )
    ]:
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                mode="lines",
                name=label
            )
        )

    if not visible_watch_phases.empty:
        figure.add_trace(
            go.Scatter(
                x=visible_watch_phases.index,
                y=visible_watch_phases["Close"],
                mode="markers",
                name="WATCH",
                marker={
                    "symbol": "circle",
                    "size": 10
                }
            )
        )

    if not visible_early_phases.empty:
        figure.add_trace(
            go.Scatter(
                x=visible_early_phases.index,
                y=visible_early_phases["Close"],
                mode="markers",
                name="EARLY",
                marker={
                    "symbol": "triangle-up",
                    "size": 14
                }
            )
        )

    latest_date = chart_data.index[-1]

    latest_close = float(
        chart_data.iloc[-1][
            "Close"
        ]
    )

    figure.add_trace(
        go.Scatter(
            x=[
                latest_date
            ],
            y=[
                latest_close
            ],
            mode="markers+text",
            name="Aktueller Stand",
            text=[
                "Aktuell"
            ],
            textposition="top center",
            marker={
                "symbol": "diamond",
                "size": 12
            }
        )
    )

    figure.add_vline(
        x=latest_date,
        line_dash="dot"
    )

    figure.update_layout(
        title="BTC-Kurs mit EMA-Linien, WATCH- und EARLY-Signalen",
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


def create_rsi_chart(
    chart_data
):
    figure = go.Figure()

    figure.add_hrect(
        y0=0,
        y1=30,
        opacity=0.12,
        line_width=0,
        annotation_text="Überverkauft"
    )

    figure.add_hrect(
        y0=70,
        y1=100,
        opacity=0.12,
        line_width=0,
        annotation_text="Überkauft"
    )

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
        annotation_text="RSI 30"
    )

    figure.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="RSI 70"
    )

    figure.add_trace(
        go.Scatter(
            x=[
                chart_data.index[-1]
            ],
            y=[
                float(
                    chart_data.iloc[-1][
                        "RSI"
                    ]
                )
            ],
            mode="markers",
            name="Aktueller RSI"
        )
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
        range=[
            0,
            100
        ]
    )

    return figure


def create_drawdown_chart(
    chart_data
):
    figure = go.Figure()

    figure.add_hrect(
        y0=-100,
        y1=-40,
        opacity=0.10,
        line_width=0,
        annotation_text="WATCH-Zone"
    )

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


def create_score_chart(
    chart_data
):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["SETUP_SCORE"],
            mode="lines",
            name="Setup-Score"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["CONFIRMATION_SCORE"],
            mode="lines",
            name="Bestätigungs-Score"
        )
    )

    figure.update_layout(
        title="Setup- und Bestätigungs-Score",
        height=350,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    figure.update_yaxes(
        range=[
            0,
            100
        ]
    )

    return figure


def create_backtest_chart(
    events,
    horizon
):
    column = f"RET_{horizon}D"

    if (
        events.empty
        or
        column not in events.columns
    ):
        return None

    chart_data = events[
        [
            "DATUM",
            column
        ]
    ].copy()

    chart_data = chart_data.dropna(
        subset=[
            column
        ]
    )

    if chart_data.empty:
        return None

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_data["DATUM"],
            y=chart_data[column],
            text=[
                f"{value:.1f} %"
                for value in chart_data[column]
            ],
            textposition="outside",
            name=f"{horizon}-Tage-Rendite"
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash"
    )

    figure.update_layout(
        title=(
            f"Rendite je EARLY-Signal nach "
            f"{horizon} Tagen"
        ),
        height=500,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    return figure


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.header(
    "Darstellung"
)

if st.sidebar.button(
    "Daten jetzt aktualisieren"
):
    st.cache_data.clear()

    st.rerun()

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

visible_watch_phases = watch_phases[
    watch_phases.index >=
    chart_data.index.min()
]

visible_early_phases = early_phases[
    early_phases.index >=
    chart_data.index.min()
]


# ============================================================
# 8. AKTUELLER MARKTSTATUS
# ============================================================

latest = analysis.iloc[-1]

latest_date = analysis.index[-1].date()

price = float(
    latest["Close"]
)

rsi = float(
    latest["RSI"]
)

drawdown = float(
    latest["DRAWDOWN"]
)

vol_ratio = float(
    latest["VOL_RATIO"]
)

setup_score = int(
    latest["SETUP_SCORE"]
)

confirmation_score = int(
    latest["CONFIRMATION_SCORE"]
)

watch_live = bool(
    latest["WATCH_SIGNAL"]
)

early_live = bool(
    latest["EARLY_SIGNAL"]
)

st.subheader(
    "Aktueller Marktstatus"
)

st.caption(
    f"Letzte vollständig ausgewertete Tageskerze: {latest_date} | "
    f"Datenquelle: {data_source} | "
    f"Automatischer Cache: {CACHE_TTL_SECONDS // 60} Minuten"
)

if fallback_reason:
    st.warning(
        "Aktuelle Yahoo-Finance-Daten konnten nicht geladen werden. "
        "Das Dashboard verwendet die hinterlegte Fallback-Datei."
    )

column_1, column_2, column_3, column_4, column_5, column_6 = (
    st.columns(6)
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
    "Setup-Score",
    f"{setup_score}/100"
)

column_6.metric(
    "Bestätigungs-Score",
    f"{confirmation_score}/100"
)

if early_live:
    st.success(
        "EARLY: Bullische Frühbestätigung aktiv. "
        "Die WATCH-Kriterien sind erfüllt und der MACD liegt "
        "über seiner Signallinie."
    )

elif watch_live:
    st.warning(
        "WATCH: Der Markt ist überverkauft und mindestens 40 % "
        "unter dem bisherigen Hoch. Die bullische Bestätigung fehlt noch."
    )

else:
    st.info(
        "Aktuell ist kein qualifiziertes bullisches Frühwarnsignal aktiv."
    )


# ============================================================
# 9. CHECKLISTE
# ============================================================

st.subheader(
    "Checkliste bis zur bullischen Frühbestätigung"
)

checklist = pd.DataFrame(
    [
        {
            "Bedingung": "Drawdown unter -40 %",
            "Rolle": "WATCH-Pflichtbedingung",
            "Status": format_check(
                drawdown < -40
            )
        },
        {
            "Bedingung": "RSI unter 30",
            "Rolle": "WATCH-Pflichtbedingung",
            "Status": format_check(
                rsi < 30
            )
        },
        {
            "Bedingung": "MACD über Signallinie",
            "Rolle": "EARLY-Pflichtbedingung",
            "Status": format_check(
                float(
                    latest["MACD"]
                )
                >
                float(
                    latest["MACD_SIGNAL"]
                )
            )
        },
        {
            "Bedingung": "Volumen mindestens 1.3x",
            "Rolle": "Zusatzinformation",
            "Status": format_check(
                vol_ratio >= 1.3
            )
        },
        {
            "Bedingung": "Kurs über EMA20",
            "Rolle": "Zusatzinformation",
            "Status": format_check(
                price >
                float(
                    latest["EMA20"]
                )
            )
        },
        {
            "Bedingung": "RSI steigt gegenüber vor 3 Tagen",
            "Rolle": "Zusatzinformation",
            "Status": format_check(
                rsi >
                float(
                    analysis.iloc[-4][
                        "RSI"
                    ]
                )
            )
        }
    ]
)

st.dataframe(
    checklist,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 10. TABS
# ============================================================

tab_charts, tab_signals, tab_backtest, tab_methodology = st.tabs(
    [
        "Charts",
        "Historische EARLY-Signale",
        "Backtest",
        "Methodik"
    ]
)


with tab_charts:
    st.plotly_chart(
        create_price_chart(
            chart_data,
            visible_watch_phases,
            visible_early_phases,
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

    st.plotly_chart(
        create_score_chart(
            chart_data
        ),
        use_container_width=True
    )


with tab_signals:
    st.subheader(
        "Historische EARLY-Signalphasen"
    )

    st.caption(
        "Mehrere eng aufeinanderfolgende Signale werden durch einen "
        f"Cooldown von {COOLDOWN_DAYS} Tagen zu einer Phase zusammengefasst."
    )

    if early_events.empty:
        st.info(
            "Keine historischen EARLY-Signalphasen vorhanden."
        )

    else:
        signal_table = early_events.copy()

        signal_table["DATUM"] = (
            signal_table["DATUM"]
            .dt.date
        )

        numeric_columns = [
            "BTC_PREIS",
            "DRAWDOWN",
            "RSI",
            "VOL_RATIO",
            "RET_30D",
            "RET_90D",
            "RET_180D"
        ]

        for column in numeric_columns:
            if column in signal_table.columns:
                signal_table[column] = (
                    signal_table[column]
                    .round(1)
                )

        st.dataframe(
            signal_table,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="EARLY-Signale als CSV herunterladen",
            data=early_events.to_csv(
                index=False
            ),
            file_name="bullish_early_signals.csv",
            mime="text/csv"
        )


with tab_backtest:
    st.subheader(
        "Backtest der historischen EARLY-Signale"
    )

    st.caption(
        "Die Stichprobe ist klein. Die Ergebnisse dienen der "
        "Exploration und sind kein belastbarer Nachweis einer "
        "profitablen Strategie."
    )

    summary = create_summary_table(
        early_events
    )

    if summary.empty:
        st.info(
            "Noch keine auswertbaren Backtest-Ergebnisse vorhanden."
        )

    else:
        horizon = st.selectbox(
            "Horizont für Detailansicht",
            HORIZONS,
            index=1,
            format_func=lambda value: f"{value} Tage"
        )

        selected_row = summary[
            summary["Horizont"] ==
            f"{horizon} Tage"
        ].iloc[0]

        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )

        metric_1.metric(
            "EARLY-Signalphasen",
            int(
                selected_row["Ereignisse"]
            )
        )

        metric_2.metric(
            "Trefferquote",
            format_percent(
                selected_row["Trefferquote"]
            )
        )

        metric_3.metric(
            "Median-Rendite",
            format_percent(
                selected_row["Median"]
            )
        )

        metric_4.metric(
            "Ø Rendite",
            format_percent(
                selected_row["Ø Rendite"]
            )
        )

        backtest_chart = create_backtest_chart(
            early_events,
            horizon
        )

        if backtest_chart is not None:
            st.plotly_chart(
                backtest_chart,
                use_container_width=True
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
            formatted_summary[column] = (
                formatted_summary[column]
                .map(
                    format_percent
                )
            )

        formatted_summary["Profit Factor"] = (
            formatted_summary["Profit Factor"]
            .map(
                lambda value:
                "-"
                if pd.isna(
                    value
                )
                else f"{value:.2f}"
            )
        )

        st.dataframe(
            formatted_summary,
            use_container_width=True,
            hide_index=True
        )


with tab_methodology:
    st.subheader(
        "Methodik und Grenzen"
    )

    st.markdown(
        """
        **WATCH** wird aktiv, wenn BTC mindestens 40 % unter seinem
        bisherigen All-Time-High liegt und der RSI unter 30 fällt.

        **EARLY** wird aktiv, wenn zusätzlich der MACD über seiner
        Signallinie liegt. Erhöhtes Volumen ist bewusst keine
        Pflichtbedingung, sondern eine Zusatzinformation.

        Der **Setup-Score** beschreibt die Stärke der Stress- und
        Bodenbildungsphase. Der **Bestätigungs-Score** beschreibt,
        ob Momentum und kurzfristige Marktstruktur bereits nach oben
        drehen.

        Das Dashboard ist ein Research-Werkzeug. Die historische
        Stichprobe ist klein. Gebühren, Slippage, Positionsgrößen,
        Steuern und alternative Datenquellen sind nicht Bestandteil
        des Modells.
        """
    )

    st.write(
        f"Datenquelle: {data_source}"
    )

    st.write(
        f"Letzte vollständig ausgewertete Tageskerze: {latest_date}"
    )

    st.write(
        f"Automatische Datenaktualisierung: Cache für "
        f"{CACHE_TTL_SECONDS // 60} Minuten"
    )

    st.write(
        "Hinweis: yfinance ist ein Open-Source-Werkzeug für "
        "Research- und Bildungszwecke. Für reale Anlageentscheidungen "
        "sollte später eine robustere Marktdatenquelle ergänzt werden."
    )