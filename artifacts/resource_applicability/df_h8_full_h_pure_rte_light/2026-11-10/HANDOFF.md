> **2026-11-10 完了更新**：全9価格標本PASS・全worker終了。保存値会計とSOURCE/input・paired G/SE照合もPASS。[最終REPORT](parallel_v1/REPORT.md) / [RESULTS.csv](parallel_v1/RESULTS.csv) / [validation_summary](parallel_v1/validation_summary.json)。固定代表G=7.577983059679597e15 RZ、保存B2の112.541倍・PFの84.8682倍。公開照合後mandatory STOP。次の研究範囲はユーザーが判断する。
>
> 以下は計算中に作成した引継ぎ履歴。現在の実行状態はACTIVE_RUN.jsonと最終REPORTを優先し、runnerや会計を重複起動しない。

# H8 Full-H pure RTE：高速化後の引継ぎ

この研究は計算中であり、最終G・比較結果は未確定。高速化の実装・技術検証と引継ぎを行った段階。現在のjob/PID・固定SOURCEは [ACTIVE_RUN.json](ACTIVE_RUN.json)、進行は `parallel_v1/price_stage.json` を読む。終わったtrialを再起動しない。

## 対象と固定条件

- Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `codex/resource-eval-h8-full-h-pure-rte`。
- Worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h8-full-h-pure-rte`。
- H8 Stage C基点 `eb07156a37b1938ed3aed95c13abbb545466c573`。H4/H6 Full-H実装 `9b83b96b83f62142c9d87858dc24a94db7ac5d61` とT2検証 `7a8bc7f132e166390c943e6b32c92cc6abe1baf4` を再利用。
- Linear H8・1.00 Å・STO-3G・保存tol-only1e-8 DF28・16 system qubit、保存state/reference/fragment順0..27。
- **T=1.6、ε=.0005、P=0、S2**。全一体項非scalar成分と全DF28をRTE化し、決定論blockはゼロ。scalarは一度。
- 固定候補 `H8FH_eeefac7accfbd90c9e86`、q32/R704/K2、s22528。候補は価格取得前に固定済み。
- λ_R=66.00904003994434、log B=.49512911077888005、Ncos=188738782 / Nsin=188738780。必要shotの計算値であり実測shotではない。
- Cohort `H8_FULL_H_PURE_RTE_LIGHT_EXPLORATION_20261110`、同じcandidate/seed namespace/index0..8のn9。新しい独立確認cohort・candidate再選抜・rare samplingなし。
- 一体項sampling/D0、再構成、有限signalとsparse全22528 step照合は親directoryの保存PASSを保持する。新SCF/DF/state/target reference生成なし。

共通部品の使い方は[並列化・高速化の再利用ガイド](../../../../docs/parallel_acceleration_reuse.md)に整理した。

## 今回の高速化

旧逐次SOURCE `cfbe8ce2c7b98524cc855e24b88e54e16bdaa7ac` と11 file hashは [execution_sources.json](execution_sources.json)。これらを編集せず、追加runner `scripts/h8_full_h_price_parallel.py` と作用検証 `scripts/h8_full_h_parallel_aer.py` を使う。

- 実測で選んだ32 threadsのAer CPU double。実資源の空きに合わせて下げる。64 threadsは人工回路で遅かったため自動的に最大値にはしない。
- 価格用native/compiled回路を変えず、検証用だけに最大3量子ビットgateの厳密Operator行列を使用し、重い再帰展開を減らす。full-register unitary・近似合成・位相fitは行わない。
- 内容を検査する既存fingerprint keyとcall-local matrix cacheを再利用。人工テストでcontrol状態・正負時間・位相・actual adjoint・定義変更後のcacheを確認した。
- Aer入力はwritable C-order配列を使用する。read-only配列とcall間identity memo再利用による人工テストの不一致を修正してから適用した。
- 作用ベクトルと計算済み状態を `PRIVATE/parallel_v1/cost/.../0_actions/` に保存する。`ACTION_COMPUTED_NOT_YET_CHECKED` はPASSの意味ではない。全7検査PASSは価格trialのresult/checksで判断する。
- 多thread Aer後のforkを避け、位相検査と残りtrajectory workerはspawn。残りworkerは同じ保存loaderでcontextを復元する。実際のRAM/idle CPUと測定peakで投入数を決め、最大8は残り科学標本数に対応する。

人工17q timingは [performance_check_parallel.json](performance_check_parallel.json)。旧4.447秒に対し新32 threadsは.861秒、約5.17倍、ベクトル差7.92e-13。**本番全体の短縮率は未確認**。
保存H8の条件付きorder2/one-body/DF/CZ・両軸native/compiled検査は [parallel_validation/parallel_validation.json](parallel_validation/parallel_validation.json) でPASS。native≤8.65e-16、compiled≤1.05e-13。これは価格標本ではない。
人工作用テスト3件PASS、多thread後のspawn検査も [parallel_spawn_check.json](parallel_spawn_check.json)。以前の人工テスト7件は対応SOURCEを保持する。

## 旧試行からの移行

旧workerには途中checkpointがなく、LOW_CONTROLのPASS後・HIGH_CONTROLの途中までの結果を次の作用計算へ直接引き継げなかった。今回の明示的な高速化指示に従い、旧記録・SOURCEを保全して、新しい技術実行領域 `parallel_v1/` と `PRIVATE/parallel_v1/` へ同じ科学標本を移す。

移行観測は [parallel_transition.json](parallel_transition.json)。旧partialは未完了であり、科学的FAIL・正常な価格標本のどちらにも数えない。新代表のeventファイルを旧保存ファイルとstreaming SHA-256で比較し、**byte一致がなければnative build前にSTOP**する。結果は `parallel_v1/representative_event_identity.json`。同じ標本の技術的再計算であり、独立標本を追加・poolしない。旧directoryのraw/STOP/結果を上書きしない。

## 再開時にCodexが行うこと

1. `/home/AbeHiromu/projects/AGENTS.md`、worktree `AGENTS.md`、現行統一指示を読む。sub-agentは禁止。
2. `ACTIVE_RUN.json` のPIDを確認し、`parallel_v1/price_stage.json`、`price_progress.json`、`price_STOP.json`、`prices.json` と `PRIVATE/parallel_v1/.../0_actions/progress.json` を読む。workerが生きている間は重複起動しない。旧PIDを新jobと取り違えない。
3. 初回両軸compile/全7検査のPASS後、runnerが残りn9へ自動進行する。価格取得後に選抜・seed変更・top-upしない。資源によりpartialなら取得済みと未取得を分ける。
4. 科学的correctness FAILなら誤った費用比較を続けず保存してSTOP。未確認をPASSにせず、tolerance・入力・科学条件を変えて自動retryしない。
5. `prices.json` の `all_workers_joined=true` とworker終了を確認した後、同じvenvで次の保存値会計を実行する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h8-full-h-pure-rte
LD_LIBRARY_PATH=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=64 PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python scripts/report_h8_full_h_pure_rte.py --technical-run parallel_v1
```

6. `parallel_v1/REPORT.md`、RESULTS/SIGNALS/paired_prices/comparisons.csv、SOURCE_INDEX、validation_summaryを確認。unique(cohort,id,index)、paired整数G、固定N、RZ/CX、SE、law確率と未観測条件付き価格null、一体項実sampling、SOURCE/input保全を確認する。元逐次partialや条件付き補助検査をnへ混入しない。
7. 保存H8 B2(m5_best,p8,q1,R32,K4,CONFIRMATION n9,G=67335317264200)、PF(B0+I,m5_best,p15,q1,EXPLORATION n1,G=89291219516568)と比較する。同一q/R/Kの制御比較ではなく、異なる構成・cohortによる点比較。最適性・統計的な方式間優位・母平均保証を主張しない。
8. この同じbranchへ成果物をcommitし、通常の `git push origin codex/resource-eval-h8-full-h-pure-rte` を使用する。設定済みHTTPS credential helperを変更・無効化せず、tokenを出力しない。remote SHAと公開主要blobを照合する。
9. 最終回答はH8の構築、一体項/D0、適格性、取得candidate/n、G比、normalization/shot/回路長の競合、追加研究の価値と未確認事項を区別し、GitHub URL/branch/commit/remote SHAを示す。その後mandatory STOP。

今回の引継ぎは最終科学報告の完了ではない。S1 Main Claim、論文構成、旧SOURCE/raw/grant/ledger/marker/STOPを変更しない。広域探索・別分子・T変更・詳細科学レビュー・論文執筆は開始しない。次の研究範囲の判断は、このpilot終了後にユーザーへ戻す。
