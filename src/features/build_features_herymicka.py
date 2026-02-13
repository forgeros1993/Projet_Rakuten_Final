from pathlib import Path
import re
import html
import pandas as pd
from bs4 import BeautifulSoup

# Si tu veux forcer l'usage du fichier traduit (s'il existe)
USE_TRANSLATIONS_IF_AVAILABLE = True

def strip_html(text: str) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def basic_clean(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_product_txt(df: pd.DataFrame) -> pd.Series:
    des = df.get("designation", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    desc = df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

    des = des.map(strip_html).map(basic_clean)
    desc = desc.map(strip_html).map(basic_clean)

    return (des + " . -//- " + desc).fillna("").astype(str)

def main():
    ROOT = Path(__file__).resolve().parents[2]  # OCT25_BMLE_RAKUTEN/
    RAW = ROOT / "data" / "raw"
    PROCESSED = ROOT / "data" / "processed"
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Fichiers bruts
    X_train_path = RAW / "X_train_update.csv"
    y_train_path = RAW / "Y_train_CVw08PX.csv"

    # Fichier déjà traduit (si tu veux l'utiliser)
    translations_path = PROCESSED / "Rak_train_translations.csv"

    # Labels
    y = pd.read_csv(y_train_path, index_col=0).iloc[:, 0]

    if USE_TRANSLATIONS_IF_AVAILABLE and translations_path.exists():
        print(f"✅ Using translations: {translations_path}")
        df = pd.read_csv(translations_path, index_col=0)

        if "product_txt_transl" not in df.columns:
            raise ValueError("Rak_train_translations.csv doit contenir la colonne 'product_txt_transl'")

        df["product_txt"] = df["product_txt_transl"].fillna("").astype(str)

        # si le label n'est pas dans le fichier traduit, on le rajoute
        if "prdtypecode" not in df.columns:
            df["prdtypecode"] = y.reindex(df.index)

    else:
        print("⚙️ No translations file found (or disabled). Building product_txt from raw.")
        X_train = pd.read_csv(X_train_path, index_col=0)
        df = pd.DataFrame(index=X_train.index)
        df["product_txt"] = build_product_txt(X_train)
        df["prdtypecode"] = y.reindex(df.index)

    # Nettoyage final anti-NaN
    df["product_txt"] = df["product_txt"].fillna("").astype(str)
    df["prdtypecode"] = df["prdtypecode"].astype(int)

    out_path = PROCESSED / "train_ready.csv"
    df[["product_txt", "prdtypecode"]].to_csv(out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
