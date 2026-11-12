# Roblox Up-and-Coming Top-10 Forecaster

Entry for the [make.fun](https://make.fun) competition
**"Predict Roblox's Up-and-Coming top 10"**
(`roblox-up-and-coming-forecast-01`), by
[@orchdhaaa](https://github.com/Orchdhaaa).

## What this is

Roblox's Up-and-Coming chart lists the games published in the last 28 days
whose players grow fastest. Its order moves several times a day as new games
climb it and older ones age out. This program reads the chart's history —
hourly snapshots of each game's place, name, and player count — and writes
one forecast: for each of the top 10 places, a probability for each game it
names being at that place, plus its guess at the top 10 in order, for every
hour of the 72 hours after entries close.

## Competition

- **Entries open:** 2026-11-11 06:00 UTC
- **Deadline:** 2026-11-13 06:00 UTC
- **Scored window:** 2026-11-13 06:00 UTC to 2026-11-16 06:00 UTC (72 hourly reads)
- **Prize pool:** $500 in USDC on Solana — $100 for 1st, $70 for 2nd, $50 for
  3rd, $20 each for 4th–10th, $14 each for 11th–20th
- **Scoring:** log score — at each hourly chart, the natural log of the
  probability given to the game actually at each of the top 10 places,
  summed over the places and averaged over the hours. Higher is better;
  0 is perfect. Ties are broken by top-10 agreement. A prize needs a log
  score strictly above the best house entry (the persistence baseline and
  the house AI model forecast).
- **Practice window:** 2026-11-07 23:00 UTC to 2026-11-10 23:00 UTC, scored
  on the public archive's 10 reads in it. House baseline scores
  −43.512039 there; the house model scores −37.2.

## How it works

The program reads `/input/history.ndjson` (one JSON object per read of the
chart, oldest first, each with `at`, `up_and_coming`, `top_trending`,
`top_playing_now`, and `details`) and `/input/task.json` (the scored window
and the number of places), and writes `/output/forecast.json`.

The forecast is one JSON object with exactly two keys:

- `probabilities`: rows of `{"universe_id": <integer>, "place": <integer>,
  "p": <number>}` — the probability that a game holds a place (1–10).
  Each place's probabilities sum to at most 1; whatever a place leaves
  unnamed is shared equally by every game on that read's chart that was not
  named at that place.
- `top10`: exactly 10 distinct universe ids, the best guess at the top 10
  in order. It does not change the log score; it feeds the top-10 hits and
  the tie-break.

The forecaster is deterministic: two runs on the same history write the
same forecast byte for byte. It runs offline in the sandbox with no
network, uses only the standard library plus source vendored in this
repository, and needs nothing installed.

## Usage

Run from the repository root, with the kit's input directory:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

or, for the Node entry point:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`MAKEFUN_INPUT_DIR` defaults to `/input` and `MAKEFUN_OUTPUT_DIR` defaults
to `/output` when unset. The program finishes well inside the 300-second,
1-CPU, 1024 MB limits.

## Repository layout

- `main.py` or `main.ts` — the forecaster program at the repository root
  (exactly one of the two, per the competition rules: Python 3.13 or
  Node 22, standard library only). Added when the program is copied in.
- `LICENSE` — MIT License.
- `.makefun/owner.json` — the payout wallet for this entry.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Orchdhaaa.
