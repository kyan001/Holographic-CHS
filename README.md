# Holographic-CHS (AIGC)

Chinese trigram FTS5 for [Hermes Agent](https://hermes-agent.nousresearch.com) memory. Drops in as a memory provider plugin.

## Quick Install

```bash
hermes plugins install kyan001/Holographic-CHS --enable
```

Then set it in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: holographic-chs
```

Restart the gateway (or CLI session):

```bash
hermes gateway restart
```

## What It Does

Patches Hermes' built-in `HolographicMemoryProvider` so FTS5 uses `tokenize='trigram'` instead of the default `unicode61` tokenizer. This makes Chinese text searchable character by character.

Without this plugin, a query like `冰黑咖啡` returns zero results against a memory containing `冰的黑咖啡` — the default tokenizer splits on whitespace/punctuation only, so Chinese text is indexed as one giant token. With trigram, both strings decompose into overlapping 3-char substrings (`冰黑咖` / `的黑咖` / `黑咖啡`) and match.

## Search Strategy

Three-phase fallback for maximum recall:

1. **FTS5 AND (default)** — exact trigram match
2. **Trigram OR expansion** — retry with OR-joined trigrams when FTS5 returns empty (≥4 char queries only). Multi-word stop words (509 entries) stripped before expansion to reduce noise
3. **LIKE fallback** — SQL `LIKE '%query%'` with category filter when both FTS5 passes return nothing

## Files

```
holographic-chs/
├── plugin.yaml       # Hermes plugin metadata
└── __init__.py       # register(ctx) + implementation
```

No dependencies beyond Hermes itself. The plugin subclasses the bundled `HolographicMemoryProvider` and applies monkey-patches at initialization time.

## Updates

```bash
hermes plugins update holographic-chs
```
