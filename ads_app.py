#!/usr/bin/env python3
"""Streamlit app for Peec.ai Ads Analyzer

Run:
  streamlit run "peec_ads_app.py"

"""
import os
import json
from urllib.parse import urlparse
from collections import Counter
import re
import runpy
import io

import requests
import pandas as pd
import streamlit as st


def extract_domain(url):
    try:
        p = urlparse(url)
        return p.netloc.lower()
    except Exception:
        return None


def tokenize(text):
    if not text:
        return []
    tokens = re.findall(r"\w{3,}", text.lower())
    stop = {"the","and","for","with","that","this","from","your","are","use","using"}
    return [t for t in tokens if t not in stop]


def get_card_detail(card_list, key):
    """Safely extracts a value from the first dictionary in a list of cards."""
    if card_list and isinstance(card_list, list) and len(card_list) > 0 and isinstance(card_list[0], dict):
        return card_list[0].get(key)
    return None


def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}


def fetch_chats(base_url, headers, params):
    resp = requests.get(f"{base_url}/customer/v1/chats", headers=headers, params=params)
    resp.raise_for_status()
    return safe_json(resp).get("data", [])


def fetch_chat_content(base_url, headers, chat_id, project_id):
    resp = requests.get(f"{base_url}/customer/v1/chats/{chat_id}/content", headers=headers, params={"project_id": project_id})
    if resp.status_code != 200:
        return None
    return safe_json(resp)


def analyze_ads_simple(all_ads, clients_df=None):
    df = pd.json_normalize(all_ads, sep="_") if all_ads else pd.DataFrame()
    summary = {"total_ads": len(df)}
    if df.empty:
        return df, summary

    # extract common card fields if present
    if "cards" in df.columns:
        df["card_title"] = df["cards"].apply(lambda x: get_card_detail(x, "title"))
        df["card_body"] = df["cards"].apply(lambda x: get_card_detail(x, "body"))
        df["card_image_url"] = df["cards"].apply(lambda x: get_card_detail(x, "imageUrl"))
        df["card_target_url"] = df["cards"].apply(lambda x: get_card_detail(x, "targetUrl"))
        df = df.drop(columns=["cards"])

    if "chat_id" in df.columns:
        summary["ads_per_chat_top"] = df.groupby("chat_id").size().sort_values(ascending=False).head(10).to_dict()

    # accounts
    account_cols = [c for c in df.columns if c.endswith("account") or c.endswith("account_id") or c == "account"]
    accounts = Counter()
    for col in account_cols:
        accounts.update(df[col].dropna().astype(str).tolist())
    summary["competitor_accounts_top"] = accounts.most_common(20)

    # landing domains
    landing_cols = [c for c in df.columns if "landing" in c or "url" in c or "link" in c]
    landing_pages = Counter()
    for col in landing_cols:
        landing_pages.update(df[col].dropna().astype(str).tolist())
    domains = Counter()
    for lp, cnt in landing_pages.items():
        dom = extract_domain(lp)
        if dom:
            domains[dom] += cnt
    summary["landing_domain_top"] = domains.most_common(20)

    # text tokens
    text_cols = [c for c in df.columns if c.endswith("text") or c.endswith("ad_text") or c == "text"]
    texts = df[text_cols].astype(str).fillna("").agg(" ".join, axis=1) if text_cols else pd.Series([""]*len(df))
    token_counter = Counter()
    for t in texts:
        token_counter.update(tokenize(t))
    summary["top_text_tokens"] = token_counter.most_common(30)

    # prompts
    prompt_cols = [c for c in df.columns if c.endswith("prompt") or c == "prompt" or "prompts" in c]
    prompts = Counter()
    if prompt_cols:
        for col in prompt_cols:
            prompts.update(df[col].dropna().astype(str).tolist())
    summary["top_prompts"] = prompts.most_common(30)

    return df, summary


def main():
    st.title("Peec.ai Ads Analyzer")
    st.markdown("Analyze ChatGPT ads extracted from Peec.ai chats and surface competitor strategies.")

    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("API key (or leave blank to use PEEC_API_KEY)")
        project_id = st.text_input("Project ID", value="or_1a89d9ec-2307-4265-9669-4e994aba70ca")
    with col2:
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")
        base_url = st.text_input("Base API URL", value="https://api.peec.ai")

    # removed optional clients CSV uploader per UI simplification

    run = st.button("Run analysis")
    if run:
        st.spinner("Fetching chats and ads...")
        key = api_key or os.environ.get("PEEC_API_KEY")
        if not key:
            st.error("API key required via input or PEEC_API_KEY env var")
            return

        headers = {"X-API-Key": key}
        params = {"project_id": project_id, "features": json.dumps(["AD"]), "limit": 1000, "start_date": start_date.isoformat(), "end_date": end_date.isoformat()}

        try:
            chats = fetch_chats(base_url, headers, params)
        except Exception as e:
            st.error(f"Error fetching chats: {e}")
            return

        st.write(f"Fetched {len(chats)} chats")

        all_ads = []
        progress = st.progress(0)
        for i, chat in enumerate(chats):
            chat_id = chat.get("id")
            if not chat_id:
                continue
            content = fetch_chat_content(base_url, headers, chat_id, project_id)
            if not content:
                continue
            ads = content.get("ads") or []
            for ad in ads:
                ad["chat_id"] = chat_id
                ad.setdefault("chat_created_at", chat.get("created_at"))
                all_ads.append(ad)
            progress.progress(int((i+1)/len(chats)*100))

        st.write(f"Total ads found: {len(all_ads)}")

        df_ads, summary = analyze_ads_simple(all_ads)

        if not df_ads.empty:
            # 1) distribution of advertisers (all accounts counted over period)
            st.subheader("Advertisers")
            # detect account-like or brand-like columns (includes brandName)
            account_cols = [c for c in df_ads.columns if c.endswith("account") or c.endswith("account_id") or c == "account" or "brand" in c.lower() or "advertiser" in c.lower()]
            if account_cols:
                vals = []
                for col in account_cols:
                    vals.extend(df_ads[col].dropna().astype(str).tolist())
                if vals:
                    acc_counts = pd.Series(vals).value_counts().reset_index()
                    acc_counts.columns = ["account","count"]
                    st.bar_chart(acc_counts.set_index("account"))
                else:
                    st.info("No advertiser values found in account columns.")
            else:
                st.info("No account-like columns found in the ads data.")

            # 2) distribution of ad unit type (try common column name variants)
            st.subheader("Ad unit type")
            ad_unit_col = None
            candidates = [c for c in df_ads.columns if any(k in c.lower() for k in ("adunit","ad_unit","unit_type","adunittype","adUnit","unitType"))]
            if candidates:
                ad_unit_col = candidates[0]
            if ad_unit_col is not None:
                counts = df_ads[ad_unit_col].fillna("(blank)").astype(str).value_counts().reset_index()
                counts.columns = ["value","count"]
                st.bar_chart(counts.set_index("value"))
            else:
                st.info("No ad unit type column found in the ads data.")

            # 3) showcase card Title and card Body
            st.subheader("Card Title and Body")
            cols_to_show = [c for c in ("card_title","card_body") if c in df_ads.columns]
            if cols_to_show:
                st.dataframe(df_ads[cols_to_show].head(200))
            else:
                st.info("No card title/body fields found in the ads data.")

            # downloads
            csv_buf = io.StringIO()
            df_ads.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode()
            st.download_button("Download ads CSV", data=csv_bytes, file_name="peec_ads_export.csv", mime="text/csv")

            # summary JSON download removed per UI simplification
        else:
            st.info("No ads found for the selected range.")


if __name__ == "__main__":
    main()
