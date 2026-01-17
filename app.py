import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trading-Analyse Scalable", layout="wide")

st.title("Trading-Analyse für Scalable Capital CSV")

st.markdown(
    """
Lade hier deine **Scalable Capital Broker-Transactions.csv** hoch.
Die App erkennt:
- Wertpapierkäufe/-verkäufe
- Sparpläne
- Ein- und Auszahlungen
- Ausschüttungen/Dividenden
"""
)

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

@st.cache_data
def load_scalable_csv(file_bytes: bytes) -> pd.DataFrame:
    data = io.BytesIO(file_bytes)
    df = pd.read_csv(
        data,
        sep=";",
        header=None,
        dtype=str
    )
    # Scalable hat bei Exporten keine Header; wir geben sie manuell vor
    df.columns = [
        "date",
        "time",
        "status",
        "reference",
        "description",
        "assetType",
        "type",
        "isin",
        "shares",
        "price",
        "amount",
        "fee",
        "tax",
        "currency",
    ]

    # Zahlenfelder in float umwandeln (deutsches Komma)
    num_cols = ["shares", "price", "amount", "fee", "tax"]
    for col in num_cols:
        df[col] = (
            df[col]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Datum/Zeit
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df["date_only"] = df["datetime"].dt.date

    # Kategorien
    def classify_row(row):
        if row["assetType"] == "Security" and row["type"] in ["Buy", "Sell"]:
            return "Trade"
        if row["assetType"] == "Security" and row["type"] == "Savings plan":
            return "Savings plan"
        if row["assetType"] == "Cash" and row["type"] == "Deposit":
            return "Deposit"
        if row["assetType"] == "Cash" and row["type"] == "Withdrawal":
            return "Withdrawal"
        if row["assetType"] == "Cash" and row["type"] == "Distribution":
            return "Distribution"
        if row["assetType"] == "Security" and row["type"] == "Security transfer":
            return "Security transfer"
        return "Other"

    df["category"] = df.apply(classify_row, axis=1)

    # Cash-Flow: amount + fee + tax (NaN -> 0)
    for col in ["amount", "fee", "tax"]:
        df[col] = df[col].fillna(0.0)
    df["cash_flow"] = df["amount"] + df["fee"] + df["tax"]

    return df


def trades_table(df: pd.DataFrame) -> pd.DataFrame:
    trades = df[df["category"] == "Trade"].copy()

    # Richtung
    trades["direction"] = trades["type"]

    # Trade-ID (vereinfach: reference, falls vorhanden, sonst Kombination)
    trades["trade_id"] = trades["reference"].fillna(
        trades["isin"] + "_" + trades["datetime"].astype(str)
    )

    return trades


def daily_pl(df: pd.DataFrame) -> pd.DataFrame:
    # Realisierte Cash-Flows
    grouped = df.groupby("date_only")["cash_flow"].sum().reset_index()
    grouped = grouped.sort_values("date_only")
    grouped["equity_curve"] = grouped["cash_flow"].cumsum()
    return grouped


if uploaded_file is None:
    st.info("Bitte zuerst eine Scalable-CSV hochladen.")
    st.stop()

df = load_scalable_csv(uploaded_file.getvalue())

st.subheader("Rohdaten-Vorschau")
st.dataframe(df.head(50))

st.markdown("---")

# Filterbereich
st.sidebar.header("Filter")
category_filter = st.sidebar.multiselect(
    "Kategorien",
    options=sorted(df["category"].unique()),
    default=sorted(df["category"].unique()),
)

date_min = df["date_only"].min()
date_max = df["date_only"].max()
date_range = st.sidebar.date_input(
    "Zeitraum",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

df_filtered = df[
    df["category"].isin(category_filter)
    & (df["date_only"] >= date_range[0])
    & (df["date_only"] <= date_range[1])
]

st.subheader("Gefilterte Buchungen")
st.dataframe(df_filtered)

st.markdown("---")

# Kennzahlen
st.subheader("Kennzahlen (Cash-Flows)")

total_cf = df_filtered["cash_flow"].sum()
deposits = df_filtered[df_filtered["category"] == "Deposit"]["cash_flow"].sum()
withdrawals = df_filtered[df_filtered["category"] == "Withdrawal"]["cash_flow"].sum()
distributions = df_filtered[df_filtered["category"] == "Distribution"]["cash_flow"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gesamt-Cash-Flow", f"{total_cf:,.2f} €")
col2.metric("Einzahlungen", f"{deposits:,.2f} €")
col3.metric("Auszahlungen", f"{withdrawals:,.2f} €")
col4.metric("Ausschüttungen", f"{distributions:,.2f} €")

st.markdown("---")

# Trades
st.subheader("Trade-Übersicht")

trades = trades_table(df_filtered)
st.dataframe(
    trades[
        [
            "datetime",
            "description",
            "isin",
            "direction",
            "shares",
            "price",
            "amount",
            "fee",
            "tax",
            "cash_flow",
        ]
    ].sort_values("datetime")
)

st.markdown("---")

# Tages-P/L und Equity-Kurve
st.subheader("Tages-P/L & Equity-Kurve (Cash-Flows)")

daily = daily_pl(df_filtered)
st.line_chart(
    daily.set_index("date_only")[["cash_flow", "equity_curve"]],
    use_container_width=True,
)
st.dataframe(daily)
