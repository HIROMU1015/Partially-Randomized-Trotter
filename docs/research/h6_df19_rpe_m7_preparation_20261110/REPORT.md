# H6 DF19 RPE m7：固定2候補の実行準備

2026-11-10 JST。**m7専用runnerと人工差分3件を準備した。実H6のT=12.8 reference・signal・bias・phase・radius・Nは未計算。** 利用者指示§11「本指示のm=7限定科学計算は、ユーザーが明示的に実行を承認した後に開始する」に従い、科学実行は別途明示承認後とする。[PLAN.json](PLAN.json)に固定候補とcall範囲、[synthetic_tests.txt](synthetic_tests.txt)に人工確認、[CHECKS.txt](CHECKS.txt)に静的照合を保存した。公開照合後mandatory STOP。

## 固定条件と今回の差分

Linear H6、隣接1.00 Å、STO-3G、保存tol-only 1e−8 DF19・weighted-Hermitian completion、12 system mode、Nα=Nβ=3・sector400、保存数値state、targetは保存DF-Hの同sector ground energy、P=0。保存H/state/snapshot、fragment順、11 signed pieceのbinary64係数・順序、負時間、tail identity、scalar、normalization、lawを変更しない。

CA=.00159360010199040 Ha、h=.1、m0..12の13round、各方式αtotal=.05、αaxis=.05/26、βtotal/stat=.4/.2 rad、prior [−9.695399206821989,−3.1355322139814668] Ha、energy shiftなし。これらは[m6 run02][M6]の保存設定を転記し、再導出・再配分・最適化しない。E_stateはreference診断だけに使用し、priorやdecoder branchを選ばない。厳密ground同定・actual implementation/law誤差・rigorousな誤差上界は未資格。

| arm | candidate ID | 固定構成 |
|---|---|---|
| PF | `5cc99f967c6a176c43fb` | Full19、4th m5_best、q_PF=8、δ=1.6 |
| PR | `47402cdbdb4fecab7a46` | prefix7、4th m5_best、R_outer=16、K=2、q_PF=8、δ=1.6 |

共通m=7、T=12.8、q_RPE=128。IDは既存`h46_long_time_signal.cid`規約によるmetadataであり科学的PASSを意味しない。m6から同じδでouter反復を4→8へ増やす固定2候補で、候補追加・差替えはしない。

PRは既存ABS_TIME_PER_OUTERの配分`[1,2,1,2,1,2,1,2,1,2,1]`を維持し、1 outer全体で16、全128 event。q_RPE=128、q_PF=8、event数128は別の量である。各pieceへ16を重複適用しない。

## 再利用するSOURCEと最小拡張

基準publicationは **`18fcd90f1614e3bf7b91bb02d8254f968e048a94`**。m6 run02の科学SOURCEは`49523b11db9f7b6fc065e506424c22732cd86e96`で、run01と14科学SOURCE file hashおよび全candidate数値欄が一致する。run01/run02は同じm6一roundとして扱う。

追加は[`scripts/run_h6_df19_rpe_m7.py`](../../../scripts/run_h6_df19_rpe_m7.py)、固定PLAN、[`tests/check_h6_df19_rpe_m7.py`](../../../tests/check_h6_df19_rpe_m7.py)。m6 runnerの構成を引き継ぎ、baseline・固定ID/time/q・stage・出力scopeだけをm7へ接続した。旧m6/D2 runner、旧guard、数値kernelは不変で、新しいsolverや数値公式はない。

既存`src/trottertracks/h6_df19_rpe_d2.py`の`make_context/reference_pair/half_step_keys/half_step/evolve/analyze`、D2 runnerの原入力読取・atomic save・資源観測、m5 schedule、finite Taylor/normalizer、D1 geometry・disk・Hoeffding・decoder政策を再利用する。旧runnerの`main()`や旧8候補campaignは呼ばない。

実行時には新runner/PLANを含むcommitと科学SOURCE/input hashを記録する。旧14 fileはbaselineの保存hash、新2 fileは実行commitのblobと照合する。原入力3 file、保存H/state、tail context SHAとprefix7の保存scalar/lambda_Rを照合し、保存400次元sector表現へ接続する。新しいSCF/DF fit/eigensolve/state生成は行わない。

共通半step指数cache、列をまとめたaction、観測した空きCPU/RAMに基づく並列化はそのまま再利用する。恣意的な資源上限や共有設定変更は追加しない。他Codexのworktree/processへ書込まない。

## 明示承認後の一括実行範囲

| 科学call | 範囲 |
|---|---|
| full-H reference | T=12.8だけ、expm_multiplyとexpmの2経路・2 state action |
| finite mean | 上記PF/PR各1、計2候補 |
| PR内部対照 | exact-tail 1候補。corrected/rawを別列で独立伝播 |
| phase/radius/error disk/十分N | 上記2候補のX/Y軸、既存geometryとHoeffding |
| m0..6、prior/horizon/α/β | 保存値の読取・転記のみ、再計算0 |
| trajectory/build/compile/価格照会・fit・予測/N×C/RPE成功率simulation | 全て0 |

raw=corrected/Bの事後代入はせず、vector closureを保存する。scalar/identityは既存の一回会計。B/log B/attenuation、finite K2、normalization margin、数値u floor1e−9、stage依存roundoff、reference経路差・位相offsetとT×保存residualを含む既存政策を維持する。数値allowanceはempiricalでありrigorousな誤差上界ではない。actualの未知実装/law誤差を0に補完せず、actual Nは未資格のまま。

reference経路差>1e−10、exact propagation norm差>1e−9、raw/corrected vector closure>1e−9、非finite、SOURCE/input/binding不一致は既存STOP条件。正のraw半径下界とβdet+βstat≤.4を満たした場合だけideal条件付き十分Nを報告する。科学的FAILも有効な結果とし、自動retry・候補変更・閾値緩和を行わない。Nは最小shot数・実測値・全RPE成功保証ではない。

作業worktree rootで、**ユーザーの明示承認後にだけ**次を実行する：

```bash
../../h6-environments/20261011/venv/bin/python -B scripts/run_h6_df19_rpe_m7.py --output-dir docs/research/h6_df19_rpe_m7_results_20261110/run01
```

fresh outputを要求し、m6の結果や過去STOPを上書きしない。今回はこのコマンドを実行していない。承認証明書・grant等は追加しない。

## 人工確認と保存coverage

人工3件、全PASS（0.197秒）。(1) m7固定候補・政策・配分の拒否条件と旧m6/D2 guard保持、(2) q8/R16/K2の非可換2-level finite meanと既存spectral経路の一致（corrected/raw差<2e−13、配分・128 event・closure）、(3) m7 q_RPE/q_PF binding、reference診断必須、actual N・C/G未知量保持を確認した。2-level eigensystemは人工比較だけで、実H6のeigensolveではない。CLIは`--help`だけ確認した。旧D2の7件・m6の3件は既存記録を再利用し、全面再実行していない。人工PASSはm7科学性能の証拠ではない。

下表は[m6保存coverage][COV]の各軸Nをそのまま転記したもので、新しい推定や誤差再配分ではない。全行が上記H6 DF19・数値state仮定・CA/h/α/βの条件付き評価である。

| m | T | PF N（各軸） | PR N（各軸） |
|---|---:|---:|---:|
| 0 | .1 | 705 | 705 |
| 1 | .2 | 705 | 708 |
| 2 | .4 | 705 | 719 |
| 3 | .8 | 705 | 765 |
| 4 | 1.6 | 705 | 721 |
| 5 | 3.2 | 705 | 737 |
| 6 | 6.4 | 705 | 772 |
| 7 | 12.8 | 未計算 | 未計算 |

coverageは現在7個の異なるroundのまま。m6 run01/run02のNや標本数を合算しない。m7評価後はPASS/FAILを区別してm0..7のcoverageを記録し、残るm8..12は未評価のまま保持する。

## 価格担当への接続と次の段階

PLANのcandidate ID、m/T/q_RPE/q_PF/δ/formula/prefix/R_outer/K、m5 formula ID、配分、H/state/snapshot binding、保存tail scalar/lambda_R、X/Y、P0を渡す。保存contextの`binding.T=.8`は入力referenceの歴史的時刻で、今回の発展時間はcandidate.T=12.8。compiler期待条件（Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional、測定付きfull Hadamard wrapper、hashはPLAN）は将来の照合用contextでありm7 compiled証拠ではない。

[並行価格設計][PRICE]は別担当として保持し、今回はCの取得・model fit・予測・N×C集計を行わない。新m7のsignal・phase・radius・N・C・GはPLANでnull。m7がPASSしても、m8..12、actual lane、全13round成功、全G・PF/PR費用順位は未資格。

次の最小科学範囲は本書のm7固定2候補・reference2経路・PR内部対照1・scalar2だけ。**ユーザー**：この限定範囲の実行を明示承認するか判断。**Codex**：承認後に一括実行し、結果・coverage・価格handoffを公開、remote SHAと主要成果物を照合後STOP。m8以降や価格・詳細科学レビューへ自動進行しない。

今回の確認は本担当自身の技術確認であり、独立科学レビューではない。旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP・mandatory STOPを保持し、今回も公開照合後**mandatory STOP**。

[M6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/18fcd90f1614e3bf7b91bb02d8254f968e048a94/docs/research/h6_df19_rpe_m6_results_20261110/run02/REPORT.md
[COV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/18fcd90f1614e3bf7b91bb02d8254f968e048a94/docs/research/h6_df19_rpe_m6_results_20261110/COVERAGE.json
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c965ff69e3619a7b2f8313a2093cde679f542b8/docs/research/h6_df19_price_design_20261110/REPORT.md
