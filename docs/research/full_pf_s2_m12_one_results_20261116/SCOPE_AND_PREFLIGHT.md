# Full PF/S2・16h/7：m12各軸1shotの固定1件（実行前の固定と照合）

2026-11-16 JST、B2（Claude Code、主agentのみ。独立レビューではない）。ユーザー指示による新規Q3最大1件。基準は `a006b8e15` の2shot適合点で、m12だけX/Y各2→1にする。他round・他幅・他公式・PR・他配分は対象外。

## 固定する1配分

```text
N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,13,1]   (m0〜m12)
allocation_id = FULL_PF_S2_16H_OVER7_M12_ONE_01
元configuration = 11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d
total view = af18d37da2ffa763739fe6f01631c85516c210b8ce8e30fd949fbfaabf4755f5
binding = 69cc82b4dfc0d3f35020e5678f8202da2e5fcbb628a52646a0271c4fbe00dca0
```

正本は [INPUTS.json](INPUTS.json)（`basis_case_source` = a006b8e15 の2shot INPUTS `cases[0]`）。m0〜m11は2shot点・4shot点と同じ。

## 費用（同じ価格規則）

価格はFull PF自身の26軸（B2 `1002f1dfea` AXIS_TERMS rows 0〜25、16直接軸＋10条件付き閉包軸）。2shot点のprice_termsからNだけを差し替え、C・価格区分・参照は変えていない（テストで一致を確認）。

| 区分 | 4shot（保存） | 2shot（保存） | 1shot（今回） |
|---|---:|---:|---:|
| m0〜m11 | 503,610,270 | 503,610,270 | 503,610,270 |
| m12（X/Y各 N×13,961,779） | 111,694,232 | 55,847,116 | 27,923,558 |
| **合計 T(N)** | **615,304,502** | **559,457,386** | **531,533,828** |

指示の531,533,828 RZと一致する。2shot点からの削減は27,923,558 RZ（約4.99%）。

## §10.5照合

- Full PF S2のm12=1は、既存の再利用catalog（b4296b110のA_SHAPE・B_PRICE、a006b8e15の2shot）に無い（`cached_record` が None）。2shot点は同じcatalogで採択条件のまま再判定し、適合を確認した（再評価なし）。
- 2026-11-16 01:25以降のremote更新は、Bのbranchが PR-Pauli LD3 VP Q3、Dが PR-DF LD7 V0、D p7 Yoshida4 粗幅で、Full PF S2の1shot結果は無い。b4296b110のFull PF M5・B_PRICEもm12=1だが、別の公式・別の列。
- 実行中のjob：自ユーザーの別Q3として、Dの `run_p7_y4_coarse_q3_recovery.py`（p7 Yoshida4、別候補）が稼働中。Full PF S2の同等jobは無い。触れずに並行して実行する。

## 引き継ぐ評価条件

a006b8e15の2shot INPUTSをそのまま使い、caseだけ差し替えた。statistical_contract・options・adopted_evaluation・kernel・accelerationは2shot点（＝b4296b110の4shot点）と同一（テストで確認）。Full PF S2の保存13round信号（carrier `06a15ee4`）、Z1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW、exact τ=(0.00159360010199040 Ha)²。

経路は2shotと同じ：Bの既存PF worker・supervisor・cache照合・結果検査（byte同一）と既存高速DP（8 DP worker／batch 8）。新しいのは、2shot caller（a006b8e15と同一bytes）を読み込む薄いcaller `scripts/run_full_pf_s2_m12_one_fixed.py` だけ。

## count-onlyと閉包

count-only上限は outcomes 3,600・verification 38,522,192・transitions 116,347,408（固定範囲 4,225・55,043,173・141,559,036 の内側）。ρ=1の閉包・質量・残余・上下界はcall内で新たに検査し、2shot点のPASSを流用しない。

## テスト

新4件＋再利用経路の既存12件＝16件PASS（[TESTS.txt](TESTS.txt)）。固定1件・全13round・両軸・価格と評価条件の一致・2shot INPUTS不変、別configuration・別lawの拒否、実際のexecute入口を人工supervisorで通して1件・新しい保存先になること、判定ごとの達成費用（適合→531,533,828に更新、不適合→559,457,386維持と「2shotが最安」の記録、INCONCLUSIVE／UNKNOWN→1shot保留で559,457,386維持）を確認した。実H4のriskとsearchは起動していない。

## 実行と判定

- caller・INPUTS・テストをcommit・pushしてから、新しい run01 に最大1回。進捗は既存の STAGES／SEGMENTS／ACCEL_PROGRESS／RESOURCE_MONITOR と LAUNCH／PROCESS_RETURN で観測する。2shotは約145秒だった。
- 有効上界≤τで適合 → Full PFの達成費用を531,533,828 RZへ更新。
- 下界>τでモデル不適合 → 559,457,386 RZを維持し、「他roundを固定し、m12をX/Y同数・各軸1shot以上とする範囲では、2shotが最安の適合点」と記録する。
- 閾値をまたぐ（INCONCLUSIVE）か数値gate不成立（UNKNOWN） → 1shotの適否は保留し、確認済み達成費用559,457,386 RZだけを維持する。
- retry・列の変更・追加callはしない。新規信号・compile・fit・他配分のQ3は0件。公開・remote照合後にmandatory STOP。
