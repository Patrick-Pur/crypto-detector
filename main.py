import pandas as pd
import yfinance as yf


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

DATA_START = "2015-01-01"
ANALYSIS_START = "2017-01-01"
COOLDOWN_DAYS = 30
HORIZONS = [30, 90, 180]


# ============================================================
# 2. BTC-DATEN LADEN
# ============================================================

btc = yf.download(
    "BTC-USD",
    start=DATA_START,
    auto_adjust=True,
    progress=False
)

# yfinance liefert teilweise MultiIndex-Spalten.
# Daraus machen wir normale Spalten: Close, High, Low, Open, Volume.
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

if btc.empty:
    raise RuntimeError(
        "Keine BTC-Daten geladen. Bitte Internetverbindung prüfen."
    )


# ============================================================
# 3. INDIKATOREN BERECHNEN
# ============================================================

# Trend
btc["EMA20"] = btc["Close"].ewm(
    span=20,
    adjust=False
).mean()

btc["EMA50"] = btc["Close"].ewm(
    span=50,
    adjust=False
).mean()

btc["EMA200"] = btc["Close"].ewm(
    span=200,
    adjust=False
).mean()

# RSI
delta = btc["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

btc["RSI"] = 100 - (
    100 / (1 + rs)
)

# MACD
ema12 = btc["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = btc["Close"].ewm(
    span=26,
    adjust=False
).mean()

btc["MACD"] = ema12 - ema26

btc["MACD_SIGNAL"] = btc["MACD"].ewm(
    span=9,
    adjust=False
).mean()

# Volumen
btc["VOL_MA20"] = btc["Volume"].rolling(20).mean()

btc["VOL_RATIO"] = (
    btc["Volume"] / btc["VOL_MA20"]
)

# Drawdown
btc["ATH"] = btc["Close"].cummax()

btc["DRAWDOWN"] = (
    btc["Close"] / btc["ATH"] - 1
) * 100

required_columns = [
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
    "DRAWDOWN"
]

btc = btc.dropna(
    subset=required_columns
)

if len(btc) < 2:
    raise RuntimeError(
        "Zu wenige Datenpunkte für die Analyse."
    )

# Letzte Kerze entfernen, da der laufende Tag noch unvollständig sein kann.
btc = btc.iloc[:-1].copy()

# Auswertung ab 2017
analysis = btc.loc[ANALYSIS_START:].copy()


# ============================================================
# 4. SIGNALLOGIK
# ============================================================

analysis["WATCH_SIGNAL"] = (
    (analysis["RSI"] < 30) &
    (analysis["DRAWDOWN"] < -40)
)

analysis["EARLY_SIGNAL"] = (
    analysis["WATCH_SIGNAL"] &
    (
        analysis["MACD"] >
        analysis["MACD_SIGNAL"]
    )
)


# ============================================================
# 5. AKTUELLEN STATUS AUSGEBEN
# ============================================================

latest = analysis.iloc[-1]

status_date = latest.name.date()

price = float(latest["Close"])
ema20 = float(latest["EMA20"])
ema50 = float(latest["EMA50"])
ema200 = float(latest["EMA200"])

rsi = float(latest["RSI"])

macd = float(latest["MACD"])
macd_signal = float(latest["MACD_SIGNAL"])

drawdown = float(latest["DRAWDOWN"])
vol_ratio = float(latest["VOL_RATIO"])

watch_live = bool(latest["WATCH_SIGNAL"])
early_live = bool(latest["EARLY_SIGNAL"])

score = 50

# Kurzfristiger Trend
if price > ema20 and ema20 > ema50:
    score += 20
elif price < ema20 and ema20 < ema50:
    score -= 20

# Langfristiger Trend
if price > ema200:
    score += 10
else:
    score -= 10

# Drawdown
if drawdown < -60:
    score += 25
elif drawdown < -45:
    score += 15
elif drawdown < -30:
    score += 5

# RSI
if rsi < 30:
    score += 15
elif rsi > 70:
    score -= 15
elif rsi > 50:
    score += 5
elif rsi < 50:
    score -= 5

# MACD
if macd > macd_signal:
    score += 15
else:
    score -= 15

# Volumen
if vol_ratio >= 1.3:
    score += 10

score = max(
    0,
    min(100, score)
)

print("\nBTC STATUS")
print("============================================================")
print(f"Letzte abgeschlossene Kerze: {status_date}")
print(f"BTC: {price:.2f} USD")
print(f"EMA20: {ema20:.2f}")
print(f"EMA50: {ema50:.2f}")
print(f"EMA200: {ema200:.2f}")
print(f"RSI: {rsi:.2f}")
print(f"MACD: {macd:.2f}")
print(f"MACD Signal: {macd_signal:.2f}")
print(f"Drawdown: {drawdown:.1f}%")
print(f"Volumen-Faktor: {vol_ratio:.2f}x")
print(f"Experimenteller Umkehr-Score: {score}/100")

if early_live:
    print("Status: EARLY - bullische Frühbestätigung")
elif watch_live:
    print("Status: WATCH - überverkauft, aber noch unbestätigt")
else:
    print("Status: Kein qualifiziertes bullisches Frühwarnsignal")

if vol_ratio >= 1.3:
    print("Volumen-Hinweis: Erhöhtes Volumen")
elif vol_ratio >= 1.0:
    print("Volumen-Hinweis: Durchschnittliches Volumen")
else:
    print("Volumen-Hinweis: Unterdurchschnittliches Volumen")


# ============================================================
# 6. SIGNALPHASEN AUSWÄHLEN
# ============================================================

def select_signal_phases(
    data,
    signal_column,
    cooldown_days
):
    candidate_dates = list(
        data.index[
            data[signal_column].fillna(False)
        ]
    )

    selected_dates = []

    for date in candidate_dates:
        if not selected_dates:
            selected_dates.append(date)
            continue

        days_since_last_signal = (
            date - selected_dates[-1]
        ).days

        if days_since_last_signal >= cooldown_days:
            selected_dates.append(date)

    return data.loc[
        selected_dates
    ].copy()


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


# ============================================================
# 7. EINZELNE EREIGNISSE AUSWERTEN
# ============================================================

def classify_volume(vol_ratio_value):
    if vol_ratio_value >= 1.3:
        return "ERHOEHT"

    if vol_ratio_value >= 1.0:
        return "NORMAL"

    return "NIEDRIG"


def calculate_event_table(
    data,
    signal_phases
):
    records = []

    for date, row in signal_phases.iterrows():
        position = data.index.get_loc(date)

        entry_price = float(
            data.iloc[position]["Close"]
        )

        record = {
            "DATUM": date,
            "BTC_PREIS": entry_price,
            "DRAWDOWN": float(row["DRAWDOWN"]),
            "RSI": float(row["RSI"]),
            "VOL_RATIO": float(row["VOL_RATIO"]),
            "VOL_LABEL": classify_volume(
                float(row["VOL_RATIO"])
            )
        }

        for horizon in HORIZONS:
            if position + horizon >= len(data):
                record[f"RET_{horizon}D"] = None
                record[f"MAE_{horizon}D"] = None
                record[f"MFE_{horizon}D"] = None
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

            performance = (
                future_price / entry_price - 1
            ) * 100

            mae = (
                float(window.min()) /
                entry_price - 1
            ) * 100

            mfe = (
                float(window.max()) /
                entry_price - 1
            ) * 100

            record[f"RET_{horizon}D"] = performance
            record[f"MAE_{horizon}D"] = mae
            record[f"MFE_{horizon}D"] = mfe

        records.append(record)

    return pd.DataFrame(records)


early_events = calculate_event_table(
    analysis,
    early_phases
)


# ============================================================
# 8. ZUSAMMENFASSUNGEN
# ============================================================

def calculate_daily_baseline(
    data,
    condition,
    horizon
):
    future_returns = (
        data["Close"].shift(-horizon) /
        data["Close"] - 1
    ) * 100

    values = future_returns[
        condition
    ].dropna()

    if values.empty:
        return None, None, 0

    return (
        float(values.mean()),
        float(values.median()),
        len(values)
    )


def print_event_summary(
    title,
    events,
    horizon
):
    print(f"\n{title}")
    print("--------------------")

    if events.empty:
        print("Keine Signalphasen vorhanden.")
        return

    column = f"RET_{horizon}D"

    results = events[column].dropna()

    if results.empty:
        print("Noch keine vollständig auswertbaren Ereignisse.")
        return

    wins = results[
        results > 0
    ]

    losses = results[
        results <= 0
    ]

    print(f"Anzahl Ereignisse: {len(results)}")
    print(f"Gewinner: {len(wins)}")
    print(f"Verlierer: {len(losses)}")

    print(
        f"Trefferquote: "
        f"{len(wins) / len(results) * 100:.1f}%"
    )

    print(
        f"Ø Rendite: "
        f"{results.mean():.1f}%"
    )

    print(
        f"Median-Rendite: "
        f"{results.median():.1f}%"
    )

    print(
        f"Beste Rendite: "
        f"{results.max():.1f}%"
    )

    print(
        f"Schlechteste Rendite: "
        f"{results.min():.1f}%"
    )

    if not wins.empty:
        print(
            f"Ø Gewinn: "
            f"{wins.mean():.1f}%"
        )

    if not losses.empty:
        print(
            f"Ø Verlust: "
            f"{abs(losses.mean()):.1f}%"
        )

    if not wins.empty and not losses.empty:
        profit_factor = (
            wins.sum() /
            abs(losses.sum())
        )

        print(
            f"Profit Factor: "
            f"{profit_factor:.2f}"
        )


# ============================================================
# 9. GESAMTER BACKTEST
# ============================================================

print("\n\nBULLISCHER EARLY-BACKTEST")
print("============================================================")
print(f"Cooldown: {COOLDOWN_DAYS} Tage")
print(f"WATCH-Phasen: {len(watch_phases)}")
print(f"EARLY-Phasen: {len(early_phases)}")

print("\nEARLY-SIGNALPHASEN")
print("--------------------")

if early_events.empty:
    print("Keine EARLY-Signalphasen vorhanden.")
else:
    table = early_events.copy()

    table["DATUM"] = table["DATUM"].dt.date

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
        table[column] = table[column].round(1)

    print(
        table[
            [
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
        ].to_string(index=False)
    )

for horizon in HORIZONS:
    print_event_summary(
        f"{horizon} Tage nach EARLY-Signal",
        early_events,
        horizon
    )


# ============================================================
# 10. BEDINGTE MARKT-BASELINES
# ============================================================

print("\n\nBEDINGTE MARKT-BASELINES")
print("============================================================")

conditions = {
    "Alle Tage": pd.Series(
        True,
        index=analysis.index
    ),

    "Drawdown < -40 %": (
        analysis["DRAWDOWN"] < -40
    ),

    "WATCH: Drawdown < -40 % und RSI < 30": (
        analysis["WATCH_SIGNAL"]
    )
}

for horizon in HORIZONS:
    print(f"\n{horizon} Tage")
    print("--------------------")

    for name, condition in conditions.items():
        mean_return, median_return, count = (
            calculate_daily_baseline(
                analysis,
                condition,
                horizon
            )
        )

        if count == 0:
            print(
                f"{name}: keine Daten"
            )
            continue

        print(
            f"{name}: "
            f"Anzahl {count} | "
            f"Ø {mean_return:.1f}% | "
            f"Median {median_return:.1f}%"
        )


# ============================================================
# 11. VIER HISTORISCHE BACKTEST-PERIODEN
# ============================================================

periods = [
    (
        "Backtest 1: 2017 bis 2019",
        "2017-01-01",
        "2019-12-31"
    ),
    (
        "Backtest 2: 2020 bis 2021",
        "2020-01-01",
        "2021-12-31"
    ),
    (
        "Backtest 3: 2022 bis 2023",
        "2022-01-01",
        "2023-12-31"
    ),
    (
        "Backtest 4: 2024 bis heute",
        "2024-01-01",
        str(analysis.index[-1].date())
    )
]

print("\n\nVIER HISTORISCHE BACKTEST-PERIODEN")
print("============================================================")

for title, start_date, end_date in periods:
    print(f"\n\n{title}")
    print("============================================================")

    if early_events.empty:
        period_events = early_events.copy()
    else:
        period_events = early_events[
            (
                early_events["DATUM"] >=
                pd.Timestamp(start_date)
            ) &
            (
                early_events["DATUM"] <=
                pd.Timestamp(end_date)
            )
        ].copy()

    print(
        f"EARLY-Signalphasen: "
        f"{len(period_events)}"
    )

    for horizon in HORIZONS:
        print_event_summary(
            f"{horizon} Tage nach EARLY-Signal",
            period_events,
            horizon
        )


# ============================================================
# 12. CSV-EXPORT
# ============================================================

signals_export_file = "bullish_early_signals.csv"
analysis_export_file = "btc_analysis.csv"

early_events.to_csv(
    signals_export_file,
    index=False
)

analysis.to_csv(
    analysis_export_file,
    index=True,
    index_label="Date"
)

print("\n\nEXPORT")
print("============================================================")

print(
    f"Einzelergebnisse gespeichert als: "
    f"{signals_export_file}"
)

print(
    f"Vollständige Chartdaten gespeichert als: "
    f"{analysis_export_file}"
)