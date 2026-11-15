# S0-r1：判定規則の修正案（提案・未実行）

run01は、固定規則（[AUDIT_RULES.md](AUDIT_RULES.md)、SOURCE `2f3bf20ce`）のP2で26件すべてFAILした。本書はその原因に対する最小の修正案である。**この修正案での再実行は、別の明示許可を要する。** run01の結果・判定は変更しない。

## 1. run01で起きたこと

- compiledのancilla CXが、固定規則の期待列より、**どの回路でもstepあたりちょうど10個少なかった**（26/26）。
- 1 path（S66/p0 X）の事後診断（[DIAGNOSTIC_P2.txt](run01/DIAGNOSTIC_P2.txt)）では、欠けた10個は、D(f1)の単一Z項5対（wire 0・3・5・6・7）だった。
  - これらのcompile前の角度は ±4.3×10⁻¹⁹ または ±8.7×10⁻¹⁹ で、数値上の0である。
- Qiskit 1.3.0は、角度の絶対値が1e−12未満のrzをidentityとして除く。その結果、間にあった `cx(anc,o)` の対が隣り合い、相殺される。
- 記録された署名から、この5対は全path・全stepで同一である。したがって、この欠落はstepごとに決まった定数である。
- 固定規則は「D領域の `cx` は期待列にすべて残る」としていたため、この相殺を扱えなかった。

## 2. 修正（3点だけ）

| # | 変更 | 理由 |
|---|---|---|
| 1 | compile前の角度の絶対値が `1e−12` 未満のD対（identity対）を、期待されるancilla CX列から除く。除いた対の数をstepごとに記録する。それ以外の照合は完全一致のまま | compilerの文書化された閾値（`ANGLE_ZERO_EPSILON`）による決定論的な除去を、期待列に反映する |
| 2 | X/Yのcompiled一致を、P2の成否と無関係に、全wireのtoken列（最終runを除く）で直接比べる | run01では区切りが失敗したため、この比較が評価されなかった |
| 3 | 区切りに成功したpathが0件のとき、P3・P6をUNKNOWN（`UNKNOWN_NO_SEGMENTED_PATH`）とする。P3のcompile前の部分（(a)(b)）は、構造の読取に成功した全pathで評価する。X/Y一致が未評価ならP5をPASSにしない | run01の集計で、空の集合に対する判定が「FAIL」や「PASS」と表示された（空虚な真・偽）。正しくは未評価である |

閾値・区切り位置・P3〜P6の判定基準・STOP条件は、ほかに変更しない。

## 3. 実装と検証

- code：`scripts/pauli_ld2_step_audit.py`（本提案のcommit）。定数 `IDENTITY_PAIR_EPS = 1e-12`、`S0R1_IDENTITY_PAIRS = True`。
- 人工test：14件すべてPASS（[TESTS_R1.txt](TESTS_R1.txt)）。
  - test 13：toyのD領域に数値0の対を入れ、Qiskit 1.3.0と同じ除去（1e−12未満のrzを除き、隣り合った同一CXを相殺）を模した。旧規則ではrun01と同じANCILLA_CX_ALIGNMENTで失敗し、不足はstepあたり2個になることを再現した。区切りに成功したpathがないときに、P3・P6がUNKNOWNになり、X/Y一致がP2と無関係に評価されることも確認した。
  - test 14：修正規則では、P2・P3・P6がPASS、P4がPASS_EXACTになり、identity対の数（stepあたり1）が記録される。
  - test 1〜12：既存の検査（失敗を注入した場合を含む）は変わらずPASS。

## 4. 再実行する場合の範囲（許可された場合だけ）

- 対象・読取：run01と同じ13 path × X/Y。QPY 52件を再び読む（新しいfileは0）。compile・sampling・build・fitは0。
- 出力：`run02/`。run01は書き換えない。
- 規則：本提案のcommitで固定し、run02の前に変更しない。

**事前固定についての注記**
- QPYはrun01で一度開いている。
- ただしrun01では区切りに成功したpathがなく、区間・境界・合成・分散のデータは計算も閲覧もしていない。
- 修正は、alignmentの診断と、compilerの文書化された閾値だけから決めた。H_locの判定結果を見て選んだものではない。
