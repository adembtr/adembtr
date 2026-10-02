"""
Builds dark_mode.svg and light_mode.svg — my GitHub profile card.

Left:  ASCII portrait (see ascii_portrait.py)
Right: neofetch-style facts + live GitHub stats pulled from the GraphQL API

Runs every day in GitHub Actions (.github/workflows/build.yaml). Locally:
    GITHUB_TOKEN=$(gh auth token) python today.py
Only public, non-fork repositories are counted. Per-repo line counts are cached in cache/loc.json and
recomputed only when a repository's default branch moves.
"""
import datetime as dt
import json
import os
import urllib.request
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
USER = os.environ.get("USER_NAME", "adembtr")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
CACHE = os.path.join(HERE, "cache", "loc.json")
ENROLLED = dt.date(2024, 8, 19)          # first day as a Computer Engineering student at Sakarya University

# ── card geometry ───────────────────────────────────────────────────────────────────────────────
W, H = 1025, 590
ART_X, ART_Y, ART_FONT, ART_LH = 15, 18.7, 4.35, 5.0   # 112 rows of tiny glyphs fill the 560 px art area
INFO_X, INFO_Y, INFO_FONT, INFO_LH, INFO_CHARS = 450, 30, 16, 20, 58
ADVANCE = 0.602                          # monospace glyph advance in em (DejaVu Sans Mono / Menlo)

THEMES = {
    "dark": dict(bg="#161b22", text="#c9d1d9", art="#c9d1d9", key="#ffa657", value="#a5d6ff",
                 cc="#616e7f", add="#3fb950", dele="#f85149"),
    "light": dict(bg="#f6f8fa", text="#24292f", art="#24292f", key="#953800", value="#0a3069",
                  cc="#c2cfde", add="#1a7f37", dele="#cf222e"),
}


# ── GitHub data ─────────────────────────────────────────────────────────────────────────────────
def graphql(query, **variables):
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


def github_stats():
    me = graphql("""query($login:String!){ user(login:$login){ id followers{ totalCount }
        repositoriesContributedTo(contributionTypes:[COMMIT,PULL_REQUEST,REPOSITORY]){ totalCount } } }""",
                 login=USER)["user"]
    repos, cursor = [], None
    while True:
        page = graphql("""query($login:String!,$id:ID!,$after:String){ user(login:$login){
            repositories(first:50, after:$after, ownerAffiliations:OWNER, privacy:PUBLIC, isFork:false){
              pageInfo{ hasNextPage endCursor }
              nodes{ name stargazerCount defaultBranchRef{ target{ ... on Commit{
                oid history(author:{id:$id}){ totalCount } } } } } } } }""",
                       login=USER, id=me["id"], after=cursor)["user"]["repositories"]
        repos += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    add = dele = commits = 0
    for r in repos:
        tgt = (r["defaultBranchRef"] or {}).get("target") or {}
        n = tgt.get("history", {}).get("totalCount", 0)
        commits += n
        c = cache.get(r["name"])
        if not c or c["oid"] != tgt.get("oid"):
            c = {"oid": tgt.get("oid"), "add": 0, "del": 0}
            after = None
            while n:
                h = graphql("""query($owner:String!,$name:String!,$id:ID!,$after:String){
                    repository(owner:$owner,name:$name){ defaultBranchRef{ target{ ... on Commit{
                      history(first:100, after:$after, author:{id:$id}){
                        pageInfo{ hasNextPage endCursor } nodes{ additions deletions } } } } } } }""",
                            owner=USER, name=r["name"], id=me["id"], after=after)
                hist = h["repository"]["defaultBranchRef"]["target"]["history"]
                for node in hist["nodes"]:
                    c["add"] += node["additions"]
                    c["del"] += node["deletions"]
                if not hist["pageInfo"]["hasNextPage"]:
                    break
                after = hist["pageInfo"]["endCursor"]
            cache[r["name"]] = c
        add += c["add"]
        dele += c["del"]
    cache = {k: v for k, v in cache.items() if k in {r["name"] for r in repos}}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w"), indent=1, sort_keys=True)
    return dict(repos=len(repos), stars=sum(r["stargazerCount"] for r in repos),
                contributed=me["repositoriesContributedTo"]["totalCount"],
                followers=me["followers"]["totalCount"], commits=commits, add=add, dele=dele)


def uptime(start, today):
    y, m, d = today.year - start.year, today.month - start.month, today.day - start.day
    if d < 0:
        m -= 1
        d += (today.replace(day=1) - dt.timedelta(days=1)).day
    if m < 0:
        y, m = y - 1, m + 12
    p = lambda n, w: f"{n} {w}{'' if n == 1 else 's'}"
    return f"{p(y, 'year')}, {p(m, 'month')}, {p(d, 'day')}"


# ── card text ───────────────────────────────────────────────────────────────────────────────────
def line(key, value):
    dots = INFO_CHARS - len(key) - len(value) - 5
    return [(". ", "cc"), (key, "key"), (":", None), (" " + "." * max(dots, 1) + " ", "cc"), (value, "value")]


def header(title):
    return [(title + " ", None), ("─" * (INFO_CHARS - len(title) - 1), "cc")]


def pair(k1, v1, k2, v2, split=33):
    left = [(". ", "cc"), (k1, "key"), (":", None), (" " + "." * max(split - len(k1) - len(v1) - 6, 1) + " ", "cc"), (v1, "value")]
    used = sum(len(t) for t, _ in left)
    rest = INFO_CHARS - used - 3
    right = [(" | ", None), (k2, "key"), (":", None), (" " + "." * max(rest - len(k2) - len(v2) - 3, 1) + " ", "cc"), (v2, "value")]
    return left + right


def loc_line(s):
    key, net = "Lines of Code", f"{s['add'] - s['dele']:,}"
    tail = f" ( {s['add']:,}++, {s['dele']:,}-- )"
    dots = INFO_CHARS - len(key) - 5 - len(net) - len(tail)
    return [(". ", "cc"), (key, "key"), (":", None), (" " + "." * max(dots, 1) + " ", "cc"), (net, "value"),
            (" ( ", None), (f"{s['add']:,}++", "add"), (", ", None), (f"{s['dele']:,}--", "dele"), (" )", None)]


def info_lines(s):
    blank = [("", None)]
    return [
        header("adem@batur"),
        line("OS", "Ubuntu Linux, Android"),
        line("Uptime.Engineering", uptime(ENROLLED, dt.date.today())),
        line("Host", "Sakarya University"),
        line("Kernel", "Computer Engineering · Control & Automation"),
        line("IDE", "VS Code, Jupyter"),
        blank,
        line("Languages.Programming", "Python, C++, Dart, JavaScript"),
        line("Languages.Computer", "SQL, HTML, CSS, LaTeX, YAML"),
        line("Languages.Real", "Turkish, English"),
        blank,
        line("AI.Vision", "PyTorch, YOLO, SAM 2, DINOv3, DPVO"),
        line("AI.Agents", "LLM agents, RAG, Whisper, Z3"),
        line("Mission", "brain-like minds for humanoid robots"),
        blank,
        header("- Achievements"),
        line("TEKNOFEST 2026 AI in Aviation", "Finalist"),
        line("TÜBİTAK 2209-B", "Principal investigator"),
        blank,
        header("- Contact"),
        line("Email", "baturadem09@gmail.com"),
        line("LinkedIn", "adem-batur-652188265"),
        line("Website", "adembatur.netlify.app"),
        blank,
        header("- GitHub Stats"),
        pair("Repos", f"{s['repos']}" + (f" {{Contributed: {s['contributed']}}}" if s["contributed"] else ""),
             "Stars", f"{s['stars']:,}"),
        pair("Commits", f"{s['commits']:,}", "Followers", f"{s['followers']:,}"),
        loc_line(s),
    ]


# ── SVG ─────────────────────────────────────────────────────────────────────────────────────────
def svg(theme, art, info):
    t = THEMES[theme]
    out = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" font-family="Consolas, \'DejaVu Sans Mono\', Menlo, \'Courier New\', monospace" '
           f'width="{W}px" height="{H}px" viewBox="0 0 {W} {H}">',
           '<style>',
           f'.key{{fill:{t["key"]}}} .value{{fill:{t["value"]}}} .cc{{fill:{t["cc"]}}} .add{{fill:{t["add"]}}} .dele{{fill:{t["dele"]}}}',
           f'text, tspan {{white-space: pre;}} .art{{fill:{t["art"]}; font-size:{ART_FONT}px}} .info{{fill:{t["text"]}; font-size:{INFO_FONT}px}}',
           '</style>',
           f'<rect width="{W}px" height="{H}px" fill="{t["bg"]}" rx="15"/>']
    art_w = max(len(a) for a in art) * ADVANCE * ART_FONT
    for i, row in enumerate(art):
        row = row.ljust(len(max(art, key=len)))
        out.append(f'<text class="art" x="{ART_X}" y="{ART_Y + i * ART_LH:.1f}" textLength="{art_w:.1f}" '
                   f'lengthAdjust="spacingAndGlyphs" xml:space="preserve">{escape(row)}</text>')
    for i, segs in enumerate(info):
        n = sum(len(s) for s, _ in segs)
        if not n:
            continue
        spans = "".join(f'<tspan class="{c}">{escape(s)}</tspan>' if c else f'<tspan>{escape(s)}</tspan>' for s, c in segs)
        out.append(f'<text class="info" x="{INFO_X}" y="{INFO_Y + i * INFO_LH}" textLength="{n * ADVANCE * INFO_FONT:.1f}" '
                   f'lengthAdjust="spacingAndGlyphs" xml:space="preserve">{spans}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    stats = github_stats()
    info = info_lines(stats)
    for theme in ("dark", "light"):
        art = open(os.path.join(HERE, f"ascii_{theme}.txt"), encoding="utf-8").read().rstrip("\n").split("\n")
        with open(os.path.join(HERE, f"{theme}_mode.svg"), "w", encoding="utf-8") as f:
            f.write(svg(theme, art, info))
    print(json.dumps(stats))
