# S0-r1 判定規則（固定。run02の実行前）

S0-r1は、run01と**同じ保存QPY 52件の再解析**である。利用者は、修正案 [AUDIT_RULES_R1_PROPOSAL.md](AUDIT_RULES_R1_PROPOSAL.md)（`28f4259f0`）の3点による実行を承認した。

- run01の判定と原結果（`run01/`、SOURCE `2f3bf20ce`）は保全し、書き換えない。
- r1の結果は `run02/` に別に記録する。
- 監査code・人工test・本規則を、同じcommitで固定してから実行する。実行後に規則を変えない。

## 1. 基準となる規則

[AUDIT_RULES.md](AUDIT_RULES.md) の§1〜§8（token定義、P1〜P6、STOP・禁止事項）をそのまま使う。変更は次の§2だけである。

## 2. run01からの変更

| # | 変更 | 出所 |
|---|---|---|
| 1 | compile前の角度の絶対値が `1e−12` 未満のD対（identity対）を、期待されるancilla CX列から除く。stepごとに `identity_pairs` として数を記録する。これ以外の照合は完全一致のまま | 修正案1 |
| 2 | X/Yのcompiled一致を、全wireのtoken列（ancillaの最終runを除く）で直接比べる。P2の成否には依存しない | 修正案2 |
| 3 | 区切りに成功したpathが0件のときは、P3・P6を `UNKNOWN_NO_SEGMENTED_PATH` とする。P3のcompile前の部分（(a) D領域の内容、(b) 障壁）は、構造の読取に成功した全pathで評価する。X/Y一致が未評価ならP5をPASSにしない | 修正案3 |
| 4 | **P2の照合は、修正後の期待列とcompiledのancilla CXのtarget列を、順序まで完全一致で比べる。** 一致しなければFAILとし、次を記録する：最初の不一致の位置、前後の期待項目と実際のtarget、差分ブロック（最大10個。種類・範囲・期待項目のstep・label） | 承認時の指示 |
| 5 | P3・P4・P6は、X・YともにP2をPASSしたpathだけで評価する | 承認時の指示 |

## 3. 実行

- 入力：`INPUTS.json`（`2f3bf20ce` で固定したもの）をそのまま使う。同じ52件のQPYとcache JSONを読取専用で開く。
- 1回だけ実行する：13 pathをprocess並列、`--run --workers 13`、続けて `--summarize`。
- run01で寸法は分かっているので、`--measure`（同じfileの余分な読取）は行わない。
- 出力：`run02/`（`PATH_*.json`、`AUDIT_RESULT.json`、`AUDIT_TABLE.csv`、`SEGMENTS.csv`、`STEP_STATS.csv`、`EDGES.json`、`FINAL_RUN.csv`、`PROGRESS.jsonl`、`RUN_ALL.json`、console・time）。token配列 `run02/work/` は再生成できるため、commitしない。

## 4. 不一致があった場合

- その箇所を上記§2-4の形で記録し、そこで終える。
- 追加の規則変更や再試行はしない。閾値・区切り方法・判定基準も変えない。
- STOP条件（Qiskitの版、QPYの読取、sha256、RZ数の不一致、workerの異常）は、AUDIT_RULES.md §8のとおりである。

## 5. 範囲外

新規のpath・sampling・build・compile、fit、価格予測、E[C]・CI・G、S0+（m9 QPY）、S1・S2、希少eventの調査。
