#!/usr/bin/env python3
"""
Behavioral test for persistent story groups.

Verifies the reposted-news workflow:
1. When multiple public sources repost the same story, the articles are
   grouped and the digest summary retains every confirmed source.
2. The similarity threshold used for grouping and the stable representative
   relationship are recorded and surface in the daily digest links (stable
   block anchors).
3. An article fetched earlier that is later replaced by a more authoritative
   version still resolves to its group from the old index (article id).

Runs fully offline: OpenRouter API calls and embedding generation are stubbed.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

# Make the repo importable regardless of how this script is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rssidian.config import Config
from rssidian.core import RSSProcessor
from rssidian.models import (
    Article,
    Feed,
    StoryGroup,
    get_db_session,
    init_db,
)

PASSED = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    PASSED.append((name, condition))
    if not condition:
        raise AssertionError(name)


# Deterministic 768-dim stub embeddings: story articles share one direction,
# unrelated articles point elsewhere.
STORY_VEC = [1.0] + [0.0] * 767
OTHER_VEC = [0.0, 1.0] + [0.0] * 766


def fake_embedding(text):
    if "GPT-6" in text or "GPT" in text:
        return list(STORY_VEC)
    return list(OTHER_VEC)


def make_config(tmpdir):
    config_path = os.path.join(tmpdir, "config.toml")
    with open(config_path, "w") as f:
        f.write(
            f"""
[annoy]
index_path = "{os.path.join(tmpdir, 'annoy.idx')}"

[feeds]
similarity_threshold = 0.65
digest_similarity_threshold = 0.6

[openrouter]
minimum_quality_tier = "B"
value_prompt_enabled = true
"""
        )
    return Config(config_path)


def add_article(db, feed, title, url, guid, days_ago, words=400):
    article = Article(
        feed_id=feed.id,
        title=title,
        url=url,
        guid=guid,
        content=f"{title}. Extended report body about {guid} with details and analysis.",
        published_at=datetime.utcnow() - timedelta(days=days_ago),
        fetched_at=datetime.utcnow() - timedelta(days=days_ago),
        processed=False,
        word_count=words,
    )
    db.add(article)
    db.commit()
    return article


def main():
    tmpdir = tempfile.mkdtemp(prefix="rssidian-story-groups-")
    config = make_config(tmpdir)
    db = init_db(os.path.join(tmpdir, "test.db"))

    techdaily = Feed(title="TechDaily", url="https://techdaily.com/feed")
    blogmirror = Feed(title="BlogMirror", url="https://blogmirror.com/feed")
    newsauthority = Feed(title="NewsAuthority", url="https://newsauthority.com/feed")
    localnews = Feed(title="LocalNews", url="https://localnews.com/feed")
    for feed in (techdaily, blogmirror, newsauthority, localnews):
        db.add(feed)
    db.commit()

    processor = RSSProcessor(config, db)

    # --- Stub external services -------------------------------------------
    captured_prompts = []

    def fake_api(prompt, model=None):
        captured_prompts.append(prompt)
        if "summarizing multiple articles" in prompt:
            return "Grouped summary: TechDaily and BlogMirror both confirm the GPT-6 launch."
        if "expert content curator" in prompt:
            return (
                "## Technology\n"
                "[GPT-6 reasoning model unveiled by OpenAI](https://newsauthority.com/gpt6) "
                "was also reported as [OpenAI launches GPT-6 with new reasoning abilities]"
                "(https://techdaily.com/gpt6)."
            )
        return "Mock summary."

    tiers_by_url = {}

    def fake_analyze_value(article):
        return tiers_by_url.get(article.url, (None, None, None))

    processor._call_openrouter_api = fake_api
    processor._generate_embedding = lambda text: fake_embedding(text)
    processor._analyze_value = fake_analyze_value

    print("\n== Run 1: two public sources repost the same story ==")
    a1 = add_article(db, techdaily, "OpenAI launches GPT-6 with new reasoning abilities",
                     "https://techdaily.com/gpt6", "guid-a1", days_ago=2)
    a2 = add_article(db, blogmirror, "OpenAI launches GPT-6 with new reasoning abilities",
                     "https://blogmirror.com/post/123", "guid-a2", days_ago=2)
    u1 = add_article(db, localnews, "Local bakery wins bread competition",
                     "https://localnews.com/bakery", "guid-u1", days_ago=1)

    tiers_by_url[a1.url] = ("B", 55, "AI")
    tiers_by_url[a2.url] = ("B", 45, "AI")
    tiers_by_url[u1.url] = ("A", 70, "Food")

    stats = processor.process_articles()
    db.expire_all()

    check("two story groups persisted (pair + singleton)", stats["story_groups"] == 2)
    check("reposted articles share one persistent group",
          a1.group_id is not None and a1.group_id == a2.group_id)
    check("unrelated article gets its own group", u1.group_id not in (None, a1.group_id))

    group = db.query(StoryGroup).filter_by(id=a1.group_id).first()
    check("group records the similarity threshold used", abs(group.similarity_threshold - 0.65) < 1e-9)
    check("representative is the higher-authority version (B/55 over B/45)",
          group.representative_article_id == a1.id)

    grouped_prompts = [p for p in captured_prompts if "summarizing multiple articles" in p]
    check("grouped summary was generated once for the pair", len(grouped_prompts) == 1)
    check("grouped prompt lists every confirmed source",
          all(s in grouped_prompts[0] for s in ("Source 1", "Source 2", "TechDaily", "BlogMirror")))
    check("grouped prompt requires retaining every confirmed source",
          "EVERY confirmed source" in grouped_prompts[0])
    check("both members carry the grouped summary",
          a1.summary == a2.summary and "TechDaily" in (a1.summary or ""))

    print("\n== Run 2: a more authoritative version arrives later ==")
    b1 = add_article(db, newsauthority, "GPT-6 reasoning model unveiled by OpenAI",
                     "https://newsauthority.com/gpt6", "guid-b1", days_ago=1)
    tiers_by_url[b1.url] = ("S", 95, "AI")

    processor.process_articles()
    db.expire_all()

    check("later authoritative article joins the existing group", b1.group_id == a1.group_id)
    group = db.query(StoryGroup).filter_by(id=a1.group_id).first()
    check("S-tier version replaces the representative",
          group.representative_article_id == b1.id)
    check("group now has 3 members", len(group.members) == 3)

    print("\n== Old index entries still resolve to the group ==")
    resolved_old = processor.resolve_story_group(a1.id)
    check("superseded representative resolves to its group",
          resolved_old is not None and resolved_old["group_id"] == a1.group_id)
    check("resolution points at the current representative",
          resolved_old["representative_article_id"] == b1.id)
    check("old article is no longer flagged representative",
          resolved_old["is_representative"] is False)
    check("all members listed", set(resolved_old["member_article_ids"]) == {a1.id, a2.id, b1.id})
    resolved_new = processor.resolve_story_group(b1.id)
    check("current representative flagged as such", resolved_new["is_representative"] is True)

    print("\n== Digest: sources, threshold and stable links ==")
    digest = processor.generate_digest(lookback_days=30)
    items = digest["summary_items"]
    gid = a1.group_id

    check("merged story headed by the representative with a stable anchor",
          f"### GPT-6 reasoning model unveiled by OpenAI ^story-group-{gid}" in items)
    check("exactly one anchored group section", items.count("^story-group-") == 1)
    check("merge threshold shown in the digest", "merged at ≥0.65 similarity" in items)
    check("all confirmed sources named in the metadata line",
          "NewsAuthority + TechDaily + BlogMirror" in items)
    check("confirmed sources list keeps every source", "*Confirmed sources (3):*" in items)
    check("every source URL retained",
          all(u in items for u in
              ("https://techdaily.com/gpt6", "https://blogmirror.com/post/123",
               "https://newsauthority.com/gpt6")))
    check("representative marked in the source list",
          "**NewsAuthority**" in items and "(representative)" in items)
    check("unrelated article rendered as a plain single story",
          "### Local bakery wins bread competition" in items
          and "### Local bakery wins bread competition ^" not in items)

    check("digest exposes persistent group metadata", len(digest["story_groups"]) == 1)
    entry = digest["story_groups"][0]
    check("group metadata: id, threshold, representative, members",
          entry["group_id"] == gid
          and abs(entry["similarity_threshold"] - 0.65) < 1e-9
          and entry["representative_article_id"] == b1.id
          and set(entry["member_article_ids"]) == {a1.id, a2.id, b1.id})

    aggregated = digest["aggregated_summary"]
    check("aggregated overview links to the stable group anchor (new title)",
          f"#^story-group-{gid}|GPT-6 reasoning model unveiled by OpenAI]]" in aggregated)
    check("aggregated overview links to the stable group anchor (old title)",
          f"#^story-group-{gid}|OpenAI launches GPT-6 with new reasoning abilities]]" in aggregated)

    print("\n== Search results carry the group id (old vector index) ==")
    results = processor.search("GPT-6 reasoning", relevance_threshold=0.3)
    check("search finds the three reposts", len(results) == 3)
    check("every hit carries its story group id",
          all(r["group_id"] == gid for r in results))

    print("\n== Legacy database migration ==")
    import sqlite3
    from sqlalchemy import create_engine, inspect as sa_inspect

    old_db_path = os.path.join(tmpdir, "legacy.db")
    conn = sqlite3.connect(old_db_path)
    conn.execute("CREATE TABLE feeds (id INTEGER PRIMARY KEY, title VARCHAR(255) NOT NULL, "
                 "url VARCHAR(512) NOT NULL UNIQUE)")
    conn.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, feed_id INTEGER NOT NULL, "
                 "title VARCHAR(512) NOT NULL, url VARCHAR(512) NOT NULL, "
                 "guid VARCHAR(512) NOT NULL UNIQUE)")
    conn.commit()
    conn.close()

    legacy_session = get_db_session(old_db_path)
    inspector = sa_inspect(create_engine(f"sqlite:///{old_db_path}"))
    columns = [c["name"] for c in inspector.get_columns("articles")]
    check("story_groups table created by migration", "story_groups" in inspector.get_table_names())
    check("articles.group_id added by migration", "group_id" in columns)
    check("articles.embedding_vector still added by migration", "embedding_vector" in columns)
    legacy_session.close()

    db.close()
    print(f"\nAll {len(PASSED)} checks passed.")


if __name__ == "__main__":
    main()
