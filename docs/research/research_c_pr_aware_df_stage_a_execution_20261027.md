# 研究C：PR-aware DF Stage A限定実行・optimizer前の技術的STOP

実行日：2026-10-27 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**初期表現・物理DF予算はPASSしたが、Baseline／Proposedとも`QUALIFIED_X0_CHANGED`により実optimizer開始前にSTOPした。18/30 logical units（16 COMPLETE／2 FAIL）、ffsim fitter entry各1回、実SciPy optimizer呼出し0回、完了fit0回、retry/resume0。** 比較に必要な終了値がなく、Stage Aは未資格の`STAGE_A_NOT_MET_OR_UNRESOLVED`として報告する。改善／非改善の数値判定は未取得である。

契約の`UNAPPROVED_PROPOSAL`は歴史的metadataとして保持した。今回の明示的な実行指示に従って一度だけ実行し、承認証明書・承認原文保存・新grantは作成していない。固定SOURCEや閾値を変更せず、STOP後の追加診断計算・再実行・Stage B/Cは行わない。

## 固定範囲と再利用

| 項目 | 実行条件 |
|---|---|
| 公開準備commit | `94d1c7b4bbed106c3991ef0a5bfd1ffde337807a` |
| 数値実装commit | `1deb62f8bf648a6aaa0399f8d4eb9417f7e1c6f7` |
| Scope | `C_PR_AWARE_DF_PAIRED_STAGE_A_V1` |
| [契約](research_c_pr_aware_df_pilot_contract_draft_20261026.json) SHA256 | `0a628d801ed75142f07ff64833ac0184fa982d979efdbe3a65a3d348d09da34c` |
| [SOURCE manifest](../../artifacts/research_c_pr_aware_df/2026-10-26/technical/SOURCE.json) SHA256 | `9cf509514742a382b10ebf4bdf958e9d18f67231a9ddeecdfdf08440e8010567` |
| 系 | 保存直鎖H4・隣接1Å・STO-3G、Nα=Nβ2、sector36 |
| 元入力SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 保存CDF8初期export SHA256 | `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| DF／変数 | 通常number CDF rank8、一般実対称J／実直交U、元frame順prefix LD2 |
| 目的 | `Fγ=f/f*+γ(Mε/M*)²`、Baselineγ0／Proposedγ.1、ε1e−8 Ha、loss floor1e−14 Ha² |
| optimizer予定 | 既存ffsim0.0.83・修正codec・L-BFGS-B、options=None、各1 fit上限 |
| 物理予算 | sector operator norm＋既存engineering guard≤0.00019875 Ha |
| 元規約 | 元g補正κ固定、核scalar一度、spin／frame順維持 |
| 出力 | `artifacts/research_c_pr_aware_df/2026-10-26/run01`（契約の日付識別子を維持） |

新たなfitter・codec・operator評価法を作らず、固定`run_research_c_pr_aware_df.py`、`c_pr_aware_df.py`、`c_pr_aware_df_pilot.py`と既存ffsim・修正codec・SectorCAR／general-K／rank assessmentをそのまま使用した。既存49 tests＋29 subtestsは準備時の人工証拠であり、今回そのPASSを実データ接続のPASSと同一視しない。

## Preflightと実行資源

SOURCE28件、保存初期export／metadata／旧SOURCE3件、元入力bytes、契約・manifest、fit/evaluation双方のpackage版と固定ffsim2sourceを照合しPASS。fresh run01が未作成であることも確認した。[evaluation／資源preflight](../../artifacts/research_c_pr_aware_df/2026-10-26/execution_preflight_20261027.json)／[fit preflight](../../artifacts/research_c_pr_aware_df/2026-10-26/fit_preflight_20261027.json)。

affinity32 CPU、空きRAM約52.77 GiB、load平均約1.21、CPU／memory pressureのavg10/60/300は0。既存resource planに従い独立2 fitを別processで起動した。4×4／36×36小行列のBLASは既存planの1 thread政策でprocess内だけ調整し、JAXのCPU並列・CPU/RAM/wallに新たな上限は設けていない。cache/tempは新worktree内のtask-local領域、既存venvはread-onlyで使用した。他job・共有設定は変更していない。

既存evaluation envのPython3.11.1／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／OpenFermion1.6.1、fit envのffsim0.0.83／JAX0.4.38など、契約の全指定packageを照合した。fit側Qiskit2.4.2はffsim依存であり、P0 compileには使用していない。

固定runnerの呼出しは一回だけである。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
  scripts/run_research_c_pr_aware_df.py \
  --contract docs/research/research_c_pr_aware_df_pilot_contract_draft_20261026.json \
  --input /tmp/prt-c1-h4-b3-execution-20261026/.external/P-H4.npz \
  --output artifacts/research_c_pr_aware_df/2026-10-26/run01 --execute \
  --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

上記は実行履歴であり再実行用の許可ではない。既存run01は上書き・resumeしない。

## 取得できた初期数値

| 指標 | 保存初期値 before | 修正codec後 initial |
|---|---:|---:|
| 実rank | 8 | 8 |
| tensor Frobenius | 6.609028983585872e−5 | 6.609028983603955e−5 |
| tensor max-entry | 1.0550030930367438e−5 | 1.0550030930422949e−5 |
| 物理sector operator norm [Ha] | 7.666027761376278e−5 | 7.666027761495033e−5 |
| numerical guard [Ha] | 3.278760734406151e−9 | 3.278760734406151e−9 |
| norm＋guard [Ha] | 7.666355637449718e−5 | 7.666355637568474e−5 |
| DF予算 | PASS | PASS |
| general-K／raw-CAR max-entry | 4.052314039881821e−15 | 2.761679773755077e−15 |

codec前後の再構成tensor同士の差のFrobenius normは保存report上2.784971319744209e−15、既存allowance2.149574788330439e−10内。J差0、その他のcodec局所guardと初期sector忠実性gateもPASSし、`initial_gate.json`はPASSを記録した。物理DF予算の数値pilotであり、厳密丸めenclosureではない。

共有scaleは`f*=2.1839632053132894e−9 Ha²`、`M*=2.2093719370722074 Ha`。これはsmooth massでありexact massではない。両fitのinitial／final diagnostic hookへ到達していないため、exact tail mass・初期／最終optimizer gradient・終了tensor／sector値は取得できていない。

## STOPの位置と原因の静的確認

両armのtracebackは、既存`c1_ffsim_corrected_codec.py`のpack内にある`np.array_equal(x, expected_x0)`の不一致を示す。実SciPy `minimize`へ入る前で、ffsimの目的関数・gradientによる最適化iterationは開始していない。`success/status/message/nit/nfev/njev`や最終J/Uを欠損値0で補わずnullとした。

接続をSOURCEだけからたどると、次の二度のparameterizationがある。

1. `prepare_worker`で元J/Uを`codec.roundtrip`し、復元J/Uとその際のx0を`initial.npz`へ保存する。
2. `fit_worker`がこの**復元済み**J/Uを既存fitterへ渡し、内部packがもう一度`codec.roundtrip`する。二回目のxと一回目のx0にbitwise一致を要求する。

今回この同一性guardがFAILした。物理tensorの忠実なround-tripと、復元後のUを再packしたパラメータのbitwise冪等性は別条件である。初期物理gateがPASSしていても、この接続はPASSとはならなかった。今回の契約外で実配列を再codecして差を測定していないため、差の大きさやbranch選択の寄与は未確定で、単なる丸め差と断定しない。

準備時の人工接続試験はこのケースを捕捉できなかった。最小の修正候補は、単一の忠実なparameterizationで資格付けたx0を実fitterへそのまま引き渡し、再packの冪等性へ依存しない接続にすることである。ただし今回は固定SOURCE・契約・retry0を守り、コード修正・閾値緩和・実データ追加診断を行っていない。

## call実績と証拠

| phase | COMPLETE | FAIL | 計上 |
|---|---:|---:|---:|
| prepare | 6 | 0 | 6 |
| 初期sector／表現／gate | 8 | 0 | 8 |
| Baseline input／fitter entry | 1 | 1 | 2 |
| Proposed input／fitter entry | 1 | 1 | 2 |
| 合計 | 16 | 2 | **18/30** |

`optimized_fitter`は両entry開始時に計上されるため2 FAILも消費済みである。実optimizer0回でも、callを取消したり未実行12枠を別SOURCEへ移したりしていない。全開始workerのSTOPを回収し、親もfailed=true・retry/resume0・mandatory_stop=trueを保存した。親`failure.json`の`first_failure=null`はsubprocess exceptionが親のcall外で発生したためで、child journalの2 FAILと実tracebackを保全している。

- [全raw・SOURCE・四つのcall journal・STOP](../../artifacts/research_c_pr_aware_df/2026-10-26/run01/)
- [未編集の実行log](../../artifacts/research_c_pr_aware_df/2026-10-26/execution_20261027.log)
- [報告用summary・未取得値null](../../artifacts/research_c_pr_aware_df/2026-10-26/execution_summary_20261027.json)
- [実行後SOURCE／入力照合](../../artifacts/research_c_pr_aware_df/2026-10-26/execution_postflight_20261027.json)

runner内計測はwall2.85347秒、親CPU1.50920秒、親ru_maxrss276024 KiB。prepare wall.73755秒／peak367848 KiB、各fit entryは約.306/.308秒で停止した。各processの原値を保存し、合算job peakやoptimizer速度として解釈しない。

## Q1〜Q4と次に必要な最小作業

1. **Exact tail massを減らしたか：未判定。** 両最終値・exact mass・削減率を取得できず、1%目標の成否も未取得。初期smooth M*をexact λへ代用しない。
2. **物理DF予算：共通初期値のみPASS。** 両終了値は存在せず、Baseline／Proposedの終了予算は未資格。
3. **Optimizer停止規則：SciPy収束ではなくx0同一性guardによる技術的STOP。** 内部iterationやgradientの良否、penaltyの効果は評価していない。
4. **Stage Bの価値：今回の証拠では判断材料が足りない。** 先にx0接続の技術修正と、一般実直交U／det±／π branchを含む人工mock試験が必要。その後、新SOURCE・fresh output・実行条件を固定し、別途許可されたStage Aで物理予算とexact massを取得するのが最小手順である。

研究C／ユーザーが修正準備の採否と後続実行の範囲を判断し、Codexは具体的な指示後に技術修正を行う。本結果はPR-aware目的の科学的NO-GOでも、Stage B/Cの正式採択でもない。PF/RTE・native回路・shots・総資源・新規性の科学レビューへは進めていない。

SOURCE28件・元input・保存初期export／metadata／SOURCE3件・固定契約／manifestは実行前後で不変。旧H2/H4/H6・研究A/B・C4・Gaussian singletonのSOURCE/raw/grant/ledger/marker/STOPは変更していない。成果公開・remote照合後mandatory STOPし、retry・resume・追加fitは行わない。
