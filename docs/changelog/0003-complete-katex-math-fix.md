# 0003 - Complete the BBP math rendering fix (two more GitHub gotchas)

_Date: 2026-06-24_

## Summary

The fix in [0002](0002-fix-katex-math-rendering.md) removed the `\big`/`\Bigg`
delimiter errors but exposed two *more* GitHub-specific quirks that still broke the
two equations in [`docs/03_bbp.md`](../03_bbp.md). This entry fixes both so the
section-3 and section-4 equations finally render as proper math.

## What was changed

- [`docs/03_bbp.md`](../03_bbp.md):
  1. Replaced `\operatorname{frac}` with `\mathrm{frac}`.
  2. Reflowed the section-4 equation so the `+` joining the HEAD and TAIL sums sits
     at the **end** of the previous line instead of the **start** of its own line.

## Why

Two independent GitHub behaviors, neither of which is a problem in a standalone
KaTeX sandbox:

1. **GitHub disallows `\operatorname`.** GitHub renders math with KaTeX configured
   against an allow-list of macros, and `\operatorname` is *not* on it — it renders
   the literal error box *"The following macros are not allowed: operatorname"*.
   `\mathrm{frac}` produces the same upright "frac" text and **is** allowed.

2. **A line that starts with `+ ` (or `- `, `* `) is parsed as a Markdown list**,
   even inside a `$$ … $$` block — which terminates the math block early and dumps
   the rest as raw text (you could see the second `\underbrace` line turn into a
   bullet point). GitHub processes Markdown *before* KaTeX, so the cure is simply to
   never begin a line with a list marker inside math. Moving the `+` to the end of
   the preceding line does it; the continuation line now starts with `\underbrace`,
   which Markdown leaves alone.

Combined with [0002](0002-fix-katex-math-rendering.md), the full list of GitHub
"`$$` math" hazards this project ran into is: stripped `\{`/`\}`/`\,`, HTML-escaped
`<`/`>`, disallowed `\operatorname`, and leading-`+`/`-` list capture. The safe
toolkit: `\mathrm{...}` for operators, `\left( … \right)` for big delimiters,
`\le`/`\ge` for inequalities, and never start a math line with `+`/`-`/`*`.

## How it ties in

Pure documentation fix — no code, build, or behavior change. The equations are the
same BBP identities implemented in [`include/bbp_kernel.h`](../../include/bbp_kernel.h).

## How to try it

Open [`docs/03_bbp.md`](../03_bbp.md) on GitHub — sections 3 ("Reducing to
frac(16^d · π)") and 4 ("The head/tail split") now render as clean equations.
