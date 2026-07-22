"""Fallback tool: search trusted medical sites when the corpus can't answer.

Restricted to an allowlist of reputable health domains. The whole premise of this
project is citable, trustworthy answers - pulling in arbitrary blog content would
throw that away, so we only search sources we'd be willing to cite.
"""
from ddgs import DDGS

# Only these domains are searched. Add to the list, don't remove the restriction.
TRUSTED_DOMAINS = [
    "nhs.uk",
    "who.int",
    "medlineplus.gov",
    "mayoclinic.org",
    "cdc.gov",
]


def search(query: str, max_results: int = 4) -> list[dict]:
    """Search the trusted domains. Returns [{title, url, snippet}, ...]."""
    # DuckDuckGo's site: operator, OR'd across the allowlist
    domain_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS)
    scoped_query = f"{query} ({domain_filter})"

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(scoped_query, max_results=max_results))
    except Exception as e:          # network down, rate limited, API changed
        print(f"[web_search] failed: {e}")
        return []

    results = []
    for r in raw:
        url = r.get("href", "")
        # Belt-and-braces: verify the domain, don't trust the search engine to obey
        if any(d in url for d in TRUSTED_DOMAINS):
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("body", ""),
            })
    return results


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What are the side effects of Ozempic?"
    for i, r in enumerate(search(q), start=1):
        print(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet'][:150]}...\n")