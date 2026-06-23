"""holographic-chs — Holographic Memory Provider with Chinese trigram FTS5 support for Hermes Agent.

Subclasses the bundled HolographicMemoryProvider and applies trigram patches
to store.py and retrieval.py after initialization. Zero extra dependencies.
"""

import logging
import re
from typing import Any

from plugins.memory.holographic import HolographicMemoryProvider

logger = logging.getLogger(__name__)


def _patch_store():
    """Patch MemoryStore's internal schema for trigram FTS5."""
    import plugins.memory.holographic.store as store_mod

    # 1) Inject trigram tokenizer into FTS5 schema
    old = "USING fts5(content, tags, content=facts, content_rowid=fact_id);"
    new = "USING fts5(content, tags, content=facts, content_rowid=fact_id, tokenize='trigram');"
    if old in store_mod._SCHEMA:
        # Fast path: exact match
        store_mod._SCHEMA = store_mod._SCHEMA.replace(old, new, 1)
    elif 'tokenize' not in store_mod._SCHEMA:
        # Fallback: regex tolerant of whitespace changes
        new_schema, n = re.subn(
            r'(USING\s+fts5\s*\([^)]*\))\s*;',
            lambda m: m.group(1).rstrip() + ", tokenize='trigram');",
            store_mod._SCHEMA,
            count=1
        )
        if n == 1:
            store_mod._SCHEMA = new_schema
        else:
            logger.warning("holographic-chs: cannot inject trigram tokenizer into schema")

    # 2) Wrap _init_db on the actual class
    original_init = store_mod.MemoryStore._init_db

    def _patched_init(self):
        original_init(self)
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts_fts'"
        ).fetchone()
        if row and 'trigram' not in row['sql']:
            self._conn.execute("DROP TABLE IF EXISTS facts_fts")
            self._conn.execute(store_mod._SCHEMA)
            self._conn.execute(
                "INSERT INTO facts_fts(rowid, content, tags) "
                "SELECT fact_id, content, tags FROM facts"
            )
            self._conn.commit()
            count = self._conn.execute(
                "SELECT COUNT(*) FROM facts_fts"
            ).fetchone()[0]
            logger.info(
                "holographic-chs: migrated %d facts to trigram FTS5", count
            )
        else:
            logger.debug("holographic-chs: FTS5 already using trigram")

    store_mod.MemoryStore._init_db = _patched_init


# ── Stop-word handling ────────────────────────────────────────────────

_CHINESE_STOP_CHARS = frozenset(
    # No single-char stop words — user decided they're too risky
)

# 509 multi-word stop words from goto456/cn_stopwords.txt
_MULTI_STOP_WORDS = frozenset([
    "与此同时", "具体地说", "具体说来", "反过来说", "另一方面",
    "如上所述", "尽管如此", "总的来看", "总的来说", "总的说来",
    "总而言之", "恰恰相反", "换句话说", "由此可见", "相对而言",
    "综上所述", "这么点儿", "这就是说", "除此之外",
    "一方面", "一转眼", "不外乎", "不尽然", "不至于",
    "与其说", "且不说", "为什么", "乃至于", "之所以",
    "于是乎", "什么样", "以至于", "先不先", "再其次",
    "再者说", "反过来", "就是了", "就是说", "怎么办",
    "怎么样", "换言之", "没奈何", "甚至于", "简言之",
    "紧接着", "自个儿", "自各儿", "莫不然", "要不是",
    "要不然", "这一来", "这么些", "这么样", "这会儿",
    "那么些", "那么样", "那会儿", "难道说",
    "一些", "一何", "一切", "一则", "一旦", "一来", "一样",
    "一般", "万一", "上下", "不仅", "不但", "不光", "不单",
    "不只", "不如", "不妨", "不尽", "不得", "不怕", "不惟",
    "不成", "不拘", "不料", "不是", "不比", "不然", "不特",
    "不独", "不管", "不若", "不论", "不过", "不问",
    "与其", "与否", "且说", "两者", "个别", "为了", "为何",
    "为止", "为此", "为着", "乃至", "之一", "之类", "乌乎",
    "也好", "也罢", "二来", "于是", "云云", "云尔", "人们",
    "人家", "什么", "介于", "仍旧", "从此", "从而", "他人",
    "他们", "以上", "以为", "以便", "以免", "以及", "以故",
    "以期", "以来", "以至", "以致", "任何", "任凭", "似的",
    "但凡", "但是", "何以", "何况", "何处", "何时", "余外",
    "作为", "你们", "使得", "例如", "依据", "依照", "便于",
    "俺们", "倘使", "倘或", "倘然", "倘若", "假使", "假如",
    "假若", "傥然", "光是", "全体", "全部", "关于", "其一",
    "其中", "其二", "其他", "其余", "其它", "其次", "兼之",
    "再则", "再有", "再者", "再说", "况且", "几时", "凡是",
    "凭借", "出于", "出来", "分别", "则甚", "别人", "别处",
    "别是", "别的", "别管", "别说", "前后", "前此", "前者",
    "加之", "加以", "即令", "即使", "即便", "即如", "即或",
    "即若", "又及", "及其", "及至", "反之", "反而", "受到",
    "另外", "另悉", "只当", "只怕", "只是", "只有", "只消",
    "只要", "只限", "叮咚", "可以", "可是", "可见", "各个",
    "各位", "各种", "各自", "同时", "后者", "向使", "向着",
    "否则", "吧哒", "呜呼", "呵呵", "呼哧", "咱们", "哈哈",
    "哎呀", "哎哟", "哪个", "哪些", "哪儿", "哪天", "哪年",
    "哪怕", "哪样", "哪边", "哪里", "哼唷", "唯有", "啪达",
    "啷当", "喔唷", "嗡嗡", "嘎登", "嘿嘿", "因为", "因了",
    "因此", "因着", "因而", "固然", "在下", "在于", "基于",
    "处在", "多么", "多少", "大家", "她们", "如上", "如下",
    "如何", "如其", "如同", "如是", "如果", "如此", "如若",
    "始而", "孰料", "孰知", "宁可", "宁愿", "宁肯", "它们",
    "对于", "对待", "对方", "对比", "尔后", "尔尔", "尚且",
    "就是", "就算", "就要", "尽管", "岂但", "已矣", "巴巴",
    "并且", "并非", "庶乎", "庶几", "开外", "开始", "归齐",
    "当地", "当然", "当着", "彼时", "彼此", "得了", "怎么",
    "怎奈", "怎样", "总之", "惟其", "慢说", "我们", "或则",
    "或是", "或曰", "或者", "截至", "所以", "所在", "所幸",
    "所有", "才能", "打从", "抑或", "按照", "据此", "接着",
    "故此", "故而", "旁人", "无宁", "无论", "既往", "既是",
    "既然", "时候", "是以", "是的", "替代", "有些", "有关",
    "有及", "有时", "有的", "朝着", "本人", "本地", "本着",
    "本身", "来着", "来自", "来说", "极了", "果然", "果真",
    "某个", "某些", "某某", "根据", "正值", "正如", "正巧",
    "正是", "此地", "此处", "此外", "此时", "此次", "此间",
    "毋宁", "每当", "比及", "比如", "比方", "沿着", "漫说",
    "然则", "然后", "然而", "照着", "犹且", "犹自", "甚且",
    "甚么", "甚或", "甚而", "甚至", "用来", "由于", "由是",
    "由此", "的确", "的话", "直到", "省得", "眨眼", "着呢",
    "矣乎", "矣哉", "竟而", "等到", "等等", "类如", "纵令",
    "纵使", "纵然", "经过", "结果", "继之", "继后", "继而",
    "罢了", "而且", "而况", "而后", "而外", "而已", "而是",
    "而言", "能否", "自从", "自后", "自家", "自己", "自打",
    "自身", "至于", "至今", "至若", "般的", "若夫", "若是",
    "若果", "若非", "莫如", "莫若", "虽则", "虽然", "虽说",
    "要不", "要么", "要是", "譬喻", "譬如", "许多", "设使",
    "设或", "设若", "诚如", "诚然", "说来", "诸位", "诸如",
    "谁人", "谁料", "谁知", "贼死", "赖以", "起见", "趁着",
    "越是", "较之", "还是", "还有", "还要",
    "这个", "这么", "这些", "这儿", "这时", "这样", "这次",
    "这般", "这边", "这里",
    "进而", "连同", "逐步", "通过", "遵循", "遵照",
    "那个", "那么", "那些", "那儿", "那时", "那样", "那般",
    "那边", "那里",
    "鄙人", "鉴于", "针对", "除了", "除外", "除开", "除非",
    "随后", "随时", "随着", "非但", "非徒", "非特", "非独",
    "顺着", "首先",
])

# Compile regex once at module load time (longest matches first)
_MULTI_STOP_RE = re.compile(
    '|'.join(re.escape(w) for w in sorted(_MULTI_STOP_WORDS, key=len, reverse=True))
)


def _strip_stop(s: str) -> str:
    """Remove stop words and stop chars from string.

    1. Multi-word stop words (regex, longest-match, C-level)
    2. Single-char stop chars (set membership)
    """
    s = _MULTI_STOP_RE.sub('', s)
    return ''.join(c for c in s if c not in _CHINESE_STOP_CHARS)


def _trigram_or_query(s: str) -> str:
    """Convert text to trigram OR query for FTS5.

    Returns raw string (not OR) if ≤ 3 chars, OR-joined trigrams otherwise.
    """
    if len(s) < 4:
        return s
    trigrams = [s[i:i + 3] for i in range(len(s) - 2)]
    return ' OR '.join(f'"{t}"' for t in trigrams)


def _like_fallback(retriever, raw: str, category, min_trust, limit) -> list:
    """Run LIKE fallback; returns [] on failure."""
    if len(raw) < 2:
        return []
    try:
        sql = (
            "SELECT f.*, 0.0 as fts_rank_raw "
            "FROM facts f "
            "WHERE f.content LIKE ? "
            "  AND f.trust_score >= ?"
        )
        params = [f'%{raw}%', min_trust]
        if category:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " ORDER BY f.trust_score DESC LIMIT ?"
        params.append(limit)

        rows = retriever.store._conn.execute(sql, params).fetchall()
        if not rows:
            return []
        max_rank = max(1e-6, float(len(rows)))
        results = []
        for row in rows:
            fact = dict(row)
            fact["fts_rank_raw"] = 0.0
            fact["fts_rank"] = 1.0 / max_rank
            results.append(fact)
        return results
    except Exception:
        return []


# ── Retrieval patch ───────────────────────────────────────────────────


def _patch_retrieval(retriever):
    """Patch FactRetriever._fts_candidates: trigram OR expansion + LIKE."""
    import plugins.memory.holographic.retrieval as ret_mod

    original_fts = ret_mod.FactRetriever._fts_candidates

    def _patched_fts(self, query, category, min_trust, limit):
        # Phase 1: FTS5 trigram — query as-is (AND semantics)
        try:
            results = original_fts(self, query, category, min_trust, limit)
        except Exception:
            results = []

        # Phase 2: If empty and 4+ chars, strip stop words + trigram OR expansion
        # Stop-word stripping removes function words so the remaining trigrams
        # are content-bearing, reducing OR noise while still fixing the "冰黑咖啡"
        # → "冰的黑咖啡" recall gap.
        if not results and len(query) >= 4:
            stripped = _strip_stop(query)
            if len(stripped) >= 4:
                or_query = _trigram_or_query(stripped)
            else:
                or_query = _trigram_or_query(query)
            if ' OR ' in or_query:
                try:
                    results = original_fts(
                        self, or_query, category, min_trust, limit
                    )
                except Exception:
                    results = []

        # Phase 3: LIKE fallback when FTS5 returns nothing
        if not results:
            like_raw = query.strip('"')
            results = _like_fallback(self, like_raw, category,
                                     min_trust, limit)

        return results

    ret_mod.FactRetriever._fts_candidates = _patched_fts


# ── Provider ────────────────────────────────────────────────────────


class CustomHolographicProvider(HolographicMemoryProvider):
    """Custom holographic memory — subclass with trigram FTS5 patches."""

    @property
    def name(self) -> str:
        return "holographic-chs"

    def initialize(self, session_id: str, **kwargs) -> None:
        # Patch must happen BEFORE super().initialize() so _init_db
        # picks up the trigram schema when MemoryStore creates the DB.
        _patch_store()
        super().initialize(session_id, **kwargs)
        _patch_retrieval(self._retriever)
        logger.info("holographic-chs: initialized (trigram patched)")


# ── Plugin entry point ──────────────────────────────────────────────


def register(ctx: Any) -> None:
    config = _load_plugin_config()
    provider = CustomHolographicProvider(config=config)
    ctx.register_memory_provider(provider)


def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    import yaml
    try:
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("hermes-memory-store", {}) or {}
    except yaml.YAMLError:
        logger.warning("holographic-chs: config.yaml 解析失败 — 使用默认设置")
    except OSError:
        logger.warning("holographic-chs: config.yaml 读取失败 — 使用默认设置")
    except Exception:
        logger.warning("holographic-chs: 读取插件配置出错 — 使用默认设置", exc_info=True)
    return {}
