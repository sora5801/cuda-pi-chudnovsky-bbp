# 0002 - Fix KaTeX math rendering in the BBP doc

_Date: 2026-06-24_

## Summary

Three display-math blocks in [`docs/03_bbp.md`](../03_bbp.md) rendered as red
"Missing or unrecognized delimiter for \big / \Bigg" errors on GitHub. The math
was valid KaTeX on its own, but GitHub's Markdown layer mangles certain sequences
*before* the math reaches KaTeX. Rewrote those blocks using GitHub-safe notation so
they render correctly.

## What was changed

- [`docs/03_bbp.md`](../03_bbp.md): rewrote the "reducing to frac(16^d·π)" and
  "head/tail split" equations.
  - Replaced fractional-part brace notation `\{ x \}` and the big delimiters
    `\big\{ … \big\}` / `\Bigg\{ … \Bigg\}` with `\operatorname{frac}(x)` and
    `\left( … \right)`.
  - Replaced the literal `< 0` in a subscript with `\le -1`.
  - Removed cosmetic thin-spaces (`\,`) from inside the math.

## Why

GitHub renders `$$ … $$` by first running the text through its Markdown processor
and *then* handing the result to KaTeX. That Markdown pass:

1. **strips the backslash from escaped braces** — `\{` becomes `{` and `\}` becomes
   `}`. So `\big\{` arrives at KaTeX as `\big{`, and `\big` *requires* a real
   delimiter character (like `(` or `\lbrace`) immediately after it — a bare `{`
   is "unrecognized", hence the error;
2. **strips the backslash from `\,`** (turning thin-spaces into stray commas); and
3. **HTML-escapes `<`** into `&lt;`, which KaTeX then prints literally.

The fix is to avoid all three triggers: use `\operatorname{frac}(...)` instead of
brace-delimited fractional parts, `\left( ... \right)` instead of `\big\{ ... \big\}`,
and `\le` / `\ge` instead of `<` / `>`. These use only backslash-plus-letters
sequences and unescaped `{ }` (for arguments), which the Markdown pass leaves alone.

## How it ties in

Pure documentation fix — no code, build, or behavior change. The corrected
equations are the same BBP identities described in
[`docs/03_bbp.md`](../03_bbp.md) and implemented in
[`include/bbp_kernel.h`](../../include/bbp_kernel.h).

## How to try it

View [`docs/03_bbp.md`](../03_bbp.md) on GitHub, sections 3 and 4 — the equations
now render as proper math instead of red error boxes.
