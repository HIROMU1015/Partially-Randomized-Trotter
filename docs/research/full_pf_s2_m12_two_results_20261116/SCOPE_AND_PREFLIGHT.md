# Full PF/S2・16h/7：m12各軸2shotの固定1件（実行前の固定と照合）

2026-11-16 JST、B2（Claude Code、主agentのみ。独立レビューではない）。ユーザー指示による新規Q3最大1件。p7で費用を下げた変更（m12を各軸4→2）を、Full PF自身の信号で評価する。他配分・他幅・他公式・PRは対象外。

## 固定する1配分

```text
N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,13,2]   (m0〜m12)
allocation_id = FULL_PF_S2_16H_OVER7_M12_TWO_01
元configuration = 11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d
total view = af18d37da2ffa763739fe6f01631c85516c210b8ce8e30fd949fbfaabf4755f5
binding = 69cc82b4dfc0d3f35020e5678f8202da2e5fcbb628a52646a0271c4fbe00dca0
```

比較元は `b4296b110` の `FULL_PF_S2_16h_over_7_B_PRICE_COST_BUDGET_3O4_01`（m12各4shot、615,304,502 RZ、保存MSE上界 約1.4957834×10⁻⁶ Ha²）。m0〜m11は比較元と同じで、m12だけ4→2にした。正本は [INPUTS.json](INPUTS.json)（`seed_case_source` = b4296b110 の INPUTS `cases[1]`）。

## 費用（Full PF自身の26軸価格）

価格は B2 `1002f1dfea` の `AXIS_TERMS.json` rows 0〜25（label `FULL_PF_S2_16h_over_7`）。比較元の price_terms と C・価格区分・参照が一致すること、元configuration・total view・physical ID・元/total候補IDとの対応を、既存の照合と同じ方法で確認した。16直接軸＋10条件付き閉包軸。最終roundのm12は条件付き閉包軸である。

| 区分 | 4shot（比較元） | 2shot（今回） |
|---|---:|---:|
| m0〜m11 | 503,610,270 | 503,610,270 |
| m12（X/Y各 N×13,961,779） | 111,694,232 | 55,847,116 |
| **合計 T(N)** | **615,304,502** | **559,457,386** |

削減額は **55,847,116 RZ**（2shot×2軸×13,961,779、比較元の約9.08%）。p7の同じ変更は39,518,412 RZの削減だった（`93ab9dbbe`、別の物理候補なので比較の参考のみ）。

## §10.5照合

- 比較元 `b4296b110` 以降のBのbranch（`codex/rpe-total-dp-acceleration-20261114`、head `cbc471c91`）の新しい結果は、すべて B0I p7（幅0.1・幅h・局所探索 m12=3・m12=2・m12=1）で、Full PFは無い。
- 2026-11-15 22:00以降に更新された他のremote branch（PR-Pauli LD3、PR-DF LD7計画、C価格bridge、A Pauli計画、D m0価格、D Full-H caller準備、ロードマップ）の docs に、Full PF S2 の m12=2 結果は無い。
- b4296b110 の Full PF S2 は N0・A_SHAPE（m12=8）・B_PRICE（m12=4）で、今回の列と異なる。Full PF Yoshida4 B_PRICE も m12=2 だが別候補・別列。
- 既存の再利用catalog（Full PF S2 の A_SHAPE・B_PRICE 原API）に今回の列は無い（`cached_record` が None）。4shot点は同じcatalogで採択条件のまま再判定でき、適合を確認した（再評価はしていない）。
- 実行中のjob：自ユーザーのQ3／PF関連processは無い（別研究の古いh8 tmuxのみ）。

## 再利用する評価条件

- statistical_contract・options・adopted_evaluation は b4296b110 の INPUTS と同一（テストで一致を確認）。Z1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW、exact τ=(0.00159360010199040 Ha)²。
- 信号は carrier `06a15ee4` の Full PF S2 保存13round（`B_SIGNAL_INPUT.json`）。再計算しない。
- Q3経路はBの既存PF worker（`run_pf6_cost_budget_q3_fixed.py`、b4296b110と同一bytes）、supervisor、結果検査（`run_p7_s2_local_shot_search.py` の `cached_record`／`validate_result`、e7d24c219と同一bytes）。新しいのは固定caseを作る薄いcaller `scripts/run_full_pf_s2_m12_two_fixed.py` だけ。

## count-onlyと閉包

count-only上限は outcomes 3,600・verification 38,523,172・transitions 116,388,368（固定範囲 4,225・55,043,173・141,559,036 の内側）。ρ=1の有限binary64閉包・質量・残余・上下界はcall内で新たに検査し、4shot点のPASSを流用しない。

## テスト

新4件＋再利用経路の既存10件＝14件PASS（[TESTS.txt](TESTS.txt)）。固定1件・全13round・両軸・価格の一致・旧INPUTS不変・評価条件の一致、別configuration・別law・価格改変の拒否、実際の execute 入口を人工supervisorで通して1件だけ・新しい保存先になること、適合時だけ達成費用を更新する集計、既存workerの原API保存とmonitor例外時の保存を確認した。実H4のriskとsearchは起動していない。

## 実行と判定

- caller・INPUTS・テストをcommit・pushしてから、新しい run01 に最大1回。h6 venv、1 job・既存8 DP worker／batch 8。進捗は既存の STAGES／SEGMENTS／ACCEL_PROGRESS／RESOURCE_MONITOR と、起動・終了記録（LAUNCH／PROCESS_RETURN）で観測する。count-onlyが同じp7 m12=2は約144秒だった。
- 有効上界≤τで適合、下界>τでモデル不適合、跨げばINCONCLUSIVE、gate不成立はUNKNOWN。**適合した場合だけ** Full PF の確認済み達成費用を559,457,386 RZへ更新し、それ以外は615,304,502 RZを維持する。
- retry・列の変更・追加callはしない。新規信号・compile・価格fit・他配分のQ3は0件。終了後、結果を公開してmandatory STOP。
