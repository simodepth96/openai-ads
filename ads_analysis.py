#!/usr/bin/env python3
"""Peec.ai Ads Analyzer

Usage:
  - Provide API key via `PEEC_API_KEY` env var or `--api-key` argument.
  - Example: `python peec_ads_analyzer.py --project-id or_... --start-date 2026-05-12 --end-date 2026-06-12`

Outputs:
  - `ads_export.csv` with all normalized ad fields
  - `ads_summary.json` with computed metrics

"""
import os
import argparse
import json
from urllib.parse import urlparse
from collections import Counter, defaultdict
import requests
import pandas as pd
import re


def load_api_key(provided_key=None):
    if provided_key:
        return provided_key
    return os.environ.get("PEEC_API_KEY")


def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}


def fetch_chats(base_url, headers, params):
    resp = requests.get(f"{base_url}/customer/v1/chats", headers=headers, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Error fetching chats: {resp.status_code} {resp.text}")
    return safe_json(resp).get("data", [])


def fetch_chat_content(base_url, headers, chat_id, project_id):
    resp = requests.get(f"{base_url}/customer/v1/chats/{chat_id}/content", headers=headers, params={"project_id": project_id})
    if resp.status_code != 200:
        return None
    return safe_json(resp)


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


def analyze_ads(all_ads, clients=None):
    df = pd.json_normalize(all_ads, sep="_") if all_ads else pd.DataFrame()

    # extract common card fields if present
    if not df.empty and "cards" in df.columns:
        df["card_title"] = df["cards"].apply(lambda x: get_card_detail(x, "title"))
        df["card_body"] = df["cards"].apply(lambda x: get_card_detail(x, "body"))
        df["card_image_url"] = df["cards"].apply(lambda x: get_card_detail(x, "imageUrl"))
        df["card_target_url"] = df["cards"].apply(lambda x: get_card_detail(x, "targetUrl"))
        df = df.drop(columns=["cards"])

    summary = {}
    summary["total_ads"] = len(df)

    if len(df) == 0:
        return df, summary

    # ads per chat
    if "chat_id" in df.columns:
        summary["ads_per_chat"] = df.groupby("chat_id").size().sort_values(ascending=False).head(20).to_dict()

    # competitor accounts (attempt common field names)
    account_cols = [c for c in df.columns if c.endswith("account") or c.endswith("account_id") or c == "account"]
    accounts = Counter()
    for col in account_cols:
        accounts.update(df[col].dropna().astype(str).tolist())
    summary["competitor_accounts_top"] = accounts.most_common(30)

    # landing pages
    landing_cols = [c for c in df.columns if "landing" in c or "url" in c or "link" in c]
    landing_pages = Counter()
    for col in landing_cols:
        landing_pages.update(df[col].dropna().astype(str).tolist())
    # normalize to domains
    domains = Counter()
    for lp, cnt in landing_pages.items():
        dom = extract_domain(lp)
        if dom:
            domains[dom] += cnt
    summary["landing_domain_top"] = domains.most_common(30)

    # text strategy
    text_cols = [c for c in df.columns if c.endswith("text") or c.endswith("ad_text") or c == "text"]
    texts = df[text_cols].astype(str).fillna("").agg(" ".join, axis=1) if text_cols else pd.Series([""]*len(df))
    token_counter = Counter()
    for t in texts:
        token_counter.update(tokenize(t))
    summary["top_text_tokens"] = token_counter.most_common(50)

    # images
    img_cols = [c for c in df.columns if "image" in c or "img" in c]
    images = []
    for col in img_cols:
        if col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            images.extend(vals)
    img_hosts = Counter()
    for im in images:
        dom = extract_domain(im)
        if dom:
            img_hosts[dom] += 1
    summary["image_hosts_top"] = img_hosts.most_common(30)
    summary["ads_with_images"] = len(images)

    # prompts associated with ads
    prompt_cols = [c for c in df.columns if c.endswith("prompt") or c == "prompt" or "prompts" in c]
    prompts = Counter()
    if prompt_cols:
        for col in prompt_cols:
            prompts.update(df[col].dropna().astype(str).tolist())
    summary["top_prompts"] = prompts.most_common(50)

    # simple gap analysis if clients provided
    if clients is not None and not clients.empty:
        client_accounts = set(clients.get("account", clients.columns[0]).astype(str).tolist())
        competitor_set = set([a for a, _ in summary["competitor_accounts_top"]])
        missing_competitors = competitor_set - client_accounts
        summary["competitors_not_in_clients"] = list(missing_competitors)[:50]

        # check landing domains clients use
        client_landing_domains = set()
        if "landing_pages" in clients.columns:
            for lp in clients["landing_pages"].dropna().astype(str):
                client_landing_domains.add(extract_domain(lp))
        # domains used by competitors
        comp_domains = set([d for d, _ in summary["landing_domain_top"]])
        summary["landing_domain_gaps"] = list(comp_domains - client_landing_domains)[:50]

    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Peec.ai Ads Analyzer")
    parser.add_argument("--api-key", help="Peec API key (or set PEEC_API_KEY env var)")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--base-url", default="https://api.peec.ai")
    parser.add_argument("--clients-csv", help="Optional CSV with client accounts and landing_pages columns for gap analysis")
    parser.add_argument("--out-prefix", default="peec_ads")
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key:
        print("API key required via --api-key or PEEC_API_KEY env var")
        return

    headers = {"X-API-Key": api_key}
    params = {"project_id": args.project_id, "features": json.dumps(["AD"]), "limit": 1000, "start_date": args.start_date, "end_date": args.end_date}

    print("Fetching chats...")
    chats = fetch_chats(args.base_url, headers, params)
    print(f"Fetched {len(chats)} chats")

    all_ads = []
    for chat in chats:
        chat_id = chat.get("id")
        if not chat_id:
            continue
        content = fetch_chat_content(args.base_url, headers, chat_id, args.project_id)
        if not content:
            continue
        ads = content.get("ads") or []
        for ad in ads:
            ad["chat_id"] = chat_id
            # keep basic chat metadata useful for analysis
            ad.setdefault("chat_created_at", chat.get("created_at"))
            all_ads.append(ad)

    print(f"Total ads found: {len(all_ads)}")

    clients_df = None
    if args.clients_csv:
        try:
            clients_df = pd.read_csv(args.clients_csv)
            print(f"Loaded {len(clients_df)} client rows from {args.clients_csv}")
        except Exception as e:
            print(f"Failed loading clients CSV: {e}")

    df_ads, summary = analyze_ads(all_ads, clients=clients_df)

    # exports
    csv_out = f"{args.out_prefix}_ads_export.csv"
    json_out = f"{args.out_prefix}_ads_summary.json"
    try:
        df_ads.to_csv(csv_out, index=False)
        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {csv_out} and {json_out}")
    except Exception as e:
        print(f"Error writing outputs: {e}")

    # print concise summary
    print(json.dumps({k: summary[k] for k in ("total_ads","competitor_accounts_top","landing_domain_top","top_prompts") if k in summary}, indent=2))


if __name__ == "__main__":
    main()
