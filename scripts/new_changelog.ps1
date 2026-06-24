# =============================================================================
#  new_changelog.ps1  --  Scaffold a per-push changelog entry
# =============================================================================
#
#  This project keeps a tradition: every time something new is pushed to GitHub,
#  a Markdown file in docs/changelog/ explains what was added and why. This script
#  creates the next numbered changelog file from a template so you never forget.
#
#  USAGE:
#    powershell -File scripts\new_changelog.ps1 -Title "Add Newton division"
#  Then edit the generated docs\changelog\NNNN-add-newton-division.md before you
#  commit and push.
# =============================================================================
param(
    [Parameter(Mandatory = $true)][string]$Title
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dir  = Join-Path $root "docs\changelog"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

# Find the next sequence number by scanning existing NNNN-*.md files.
$existing = Get-ChildItem $dir -Filter "*.md" -ErrorAction SilentlyContinue |
    ForEach-Object { if ($_.Name -match '^(\d{4})-') { [int]$Matches[1] } }
$next = (($existing | Measure-Object -Maximum).Maximum) + 1
if (-not $next) { $next = 1 }
$seq = "{0:D4}" -f $next

# Slugify the title for the filename.
$slug = ($Title.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
$file = Join-Path $dir "$seq-$slug.md"

$body = @"
# $seq - $Title

_Date: $(Get-Date -Format 'yyyy-MM-dd')_

## Summary

<!-- One or two sentences: what changed and why it matters. -->

## What was added / changed

- <!-- bullet points of the concrete changes (files, features) -->

## Why

<!-- The reasoning / motivation behind the change. -->

## How it ties in

<!-- How this fits with the rest of the project (Chudnovsky, BBP, NTT, build, etc.) -->

## How to try it

```powershell
# commands to exercise the new behavior
```
"@

Set-Content -Path $file -Value $body -Encoding UTF8
Write-Host "Created $file" -ForegroundColor Green
