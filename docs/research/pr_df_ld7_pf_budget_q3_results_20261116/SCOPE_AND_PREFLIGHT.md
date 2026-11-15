# PR-DF LD7：PF達成費用を予算とする有限shot配分（実行前の固定と照合）

2026-11-16 JST、B2（Claude Code、主agentのみ。独立レビューではない）。ユーザー指示による新規Q3は最大4件。新規信号・sampling・compile・価格再fitは0件。

## 対象と入力

- 候補：PR-DF LD7（保存H4 linear 1.00 Å／STO-3G／tol-only DF10、one-bodyは別head、DF0..6 head・DF7..9 tail、S2・Δ=0.1・K2・共通κ、全13round R=1）。configuration `ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1`。
- 信号・評価契約：D2 `0defa286c` の `docs/research/d_pr_df_ld7_v0_results_20261116/Q3_INPUTS.json` の `inputs` をそのまま使った。中身は以下で、caseはV0 caseを複製し、allocationだけを差し替えた。
  - baseline case（LD7自身の保存13round信号・canonical binding・target）
  - statistical_contract・options・adopted_evaluation・acceleration
- 価格：D2 `4326422a4` の `docs/research/d_pr_df_ld7_price_results_20261116/AXIS_PRICES.json`。LD7自身の26軸参照価格で、`C_ref=α+βq`、X/Y同値、m7〜m12はq64より先の外挿。configuration・physical ID・q・R・lawとの対応を照合した。
- 保存配分・費用・Q3の索引：D2 `fd010b111` の `C_HANDOFF.json`。V0・A_SHAPE・B_PRICEの3列を、同じ価格で費用を再計算して一致を確認し、再利用catalogとした。
- 生成規則：LD7の既存規則と同じく、`e094b2d60` の有理数breakpoint `generate` を使った。各軸の上限はV0 `[64,60,56,52,47,43,38,34,30,25,21,17,12]`、重みはA_SHAPEがV0、B_PRICEがV0/C_ref、各軸1≤N≤V0。D2の保存menu（3/4予算のA/B）を同じ関数・価格で再現できることをテストで確認した。

## 固定した4列（費用の安い順）

予算は p7 S2 = 395,962,678 RZ と Full PF S2 = 559,457,386 RZ。X/Yは別々に生成し、結果として同じ列になった。4列は互いに異なり、保存3列とも一致しない。

| 順 | 予算・規則 | N_X = N_Y（m0〜m12） | exact費用 RZ | 表示 | 予算残額 RZ | m7〜m12割合 |
|---:|---|---|---|---:|---:|---:|
| 1 | p7・A_SHAPE | [15,14,13,12,11,10,9,8,7,6,5,4,2] | 704871771/2 | 352,435,885.500000 | 43,526,792.500000 | 95.5767% |
| 2 | p7・B_PRICE | [64,60,56,52,47,43,38,34,26,10,4,1,1] | 24616794490/63 | 390,742,769.682540 | 5,219,908.317460 | 82.9922% |
| 3 | Full PF・A_SHAPE | [22,21,19,18,16,15,13,11,10,8,7,5,4] | 22577997965/42 | 537,571,380.119048 | 21,886,005.880952 | 95.7475% |
| 4 | Full PF・B_PRICE | [64,60,56,52,47,43,38,34,30,19,7,3,1] | 34533437597/63 | 548,149,803.126984 | 11,307,582.873016 | 87.8761% |

各列の26軸の N×C_ref、m0〜m6／m7〜m12小計、生成のλ・次のbreakpoint費用は [INPUTS.json](INPUTS.json) にある。

## §10.5照合

- 保存LD7の結果はV0（`0defa286c`）とA_SHAPE・B_PRICE（`2b3b3bcc9`、3/4予算）の3点だけで、いずれも今回の4列と異なる（`cached_record` が全てNone）。同一列なら保存APIを採択条件で再判定して再利用する経路は、保存A_SHAPE列でテストした（適合を再現、新規call 0）。
- 2026-11-16 01:50以降のremote更新：
  - Bの `pauli_ld3_pf_budget_q3`：同じ2予算だが、PR-Pauli LD3という別候補。
  - Cの保存費用catalog：LD7は保存A_SHAPE最小点だけ。
  - D2：LD9の信号・価格。
  - いずれもLD7のPF予算列は無い。
- 実行中のjob：別チャットの D2 LD9 m6価格取得、D p7 Yoshida4較正だけ。LD7のQ3は無い。触れずに並行して実行する。

## 経路とcount-only

- Q3経路はD2のLD7と同じ：`run_pr_df_ld_split_q3_resume.py`（D2 head `460d77d8f` と同一bytes）の `supervise`／`U.worker`、固定SOURCE view `27e70efe5`、元AST guardのconstructor、Python 3.11.0rc1（h6-environments/20261011）、既存8 DP worker／batch 8。runtime記録はINPUTSに固定し、実行時に一致を確認する。
- 結果検査は既存の `validate_result`（`cbc471c91` の `run_p7_s2_local_shot_search.py` から関数を固定読込）。閉包・質量・残余・上下界・moments・原API一致を確認する。保存LD7 A_SHAPEの出力で検査が通ることをテストで確認した。
- count-only上限（D2の `bounds`）：
  - 列1：outcomes 256・verification 207,993・transitions 8,200,704。
  - 列2：4,225・53,661,931・123,742,762。
  - 列3：529・886,362・16,419,381。
  - 列4：4,225・54,265,869・128,346,666。
  - いずれも固定範囲（4,225・55,043,173・141,559,036）の内側。閉包はcall内で新たに検査する。

## テスト

新10件と、D2の再利用経路の既存2件（実際のworker pool・constructor・人工evaluator・保存）、計12件がPASSした（[TESTS.txt](TESTS.txt)）。新10件で確認した内容：
- 4列のベクトル・exact費用・残額・順序・上限・評価条件の一致。
- 生成関数がD2の保存menuを再現すること。
- 保存列の再利用、および別binding・別lawの拒否。
- 逐次規則：適合で停止、INCONCLUSIVEは保留として継続、UNKNOWNで停止、全不適合なら探索を尽くす。
- 保存LD7出力での検査関数。
- 実際の execute 入口が1件ずつ逐次に進み、最初の適合で止まること。
- 集計：適合点が無ければ1,235,340,731.912698 RZを維持し、あれば両PFと比較する。

実H4のQ3は起動していない。

## 実行と判定

- caller・INPUTS・テストをcommit・pushしてから、run01で費用の安い順に1件ずつQ3を実行する（各1回、最大4件）。進捗は既存の STAGES／ACCEL_PROGRESS／RESOURCE_MONITOR と LAUNCH／PROCESS_RETURN で観測する。D2のLD7 A/Bは1件あたり約150〜222秒だった。
- 判定は exact τ=(0.00159360010199040 Ha)² による：
  - 有効上界≤τ：適合。
  - 下界>τ：モデル不適合。
  - 区間がτをまたぐ：INCONCLUSIVE（保留として明示し、次の列へ進む）。
  - gate不成立：UNKNOWN（停止）。
- 最初の適合点で停止する。それより安い列がすべて不適合なら、生成集合内の最安適合点として記録する。適合点が無ければ、LD7の確認済み最安はA_SHAPEの1,235,340,731.912698 RZのまま。
- retry・列の変更・追加callはしない。公開・remote照合後にmandatory STOP。
