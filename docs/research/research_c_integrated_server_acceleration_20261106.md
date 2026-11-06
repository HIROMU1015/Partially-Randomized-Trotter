# 研究C：サーバー向け完全行列評価の高速化

2026-11-06 JST。技術実装・人工/回帰検証まで完了。**新しい実分子科学実行は未開始**。
旧runのresume/retryはしない。重要科学レビュー・新surrogate・H6拡張・論文化判断は含まない。

固定参照は `a7face80e10fd74e1a49dc530409db5b7f677aaa`。
専用worktreeは `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106`、
branchは `codex/research-c-integrated-server-20261106`。変更は未commitであり、実行コードは新SOURCEのファイルhashで固定する。

## 成果物

- [新SOURCE](../../artifacts/research_c_integrated/2026-11-06/server_technical/SOURCE.json)
- [サーバー向け契約](../../artifacts/research_c_integrated/2026-11-06/server_technical/contract.json)
- [passive preflight・環境・資源](../../artifacts/research_c_integrated/2026-11-06/server_technical/preflight.json)
- [人工回路の測定値](../../artifacts/research_c_integrated/2026-11-06/server_technical/artificial_benchmark.json)
- [人工/回帰テスト記録](../../artifacts/research_c_integrated/2026-11-06/server_technical/verification.json)

SOURCE SHA256は `200418968aa9e60e5ce842425e0373a4d8fafdb4f78c53a22df37b204989afd6`。
契約SHA256は `7615803fd047e19ec052b4ff39f5b1259778aa9371add1ef0db9e11e7fb06c69`。
この記録は科学SOURCEの来歴であり、ユーザーの許可を証明・代行するものではない。

## 採用した実装

`c1_h6_state_action.simulation_circuit()` の既存ゲート変換を再利用し、
`c_full_unitary.decode()` からAerのCPU/double/unitary評価器へ渡す。
最大3量子ビットのQiskit行列と公式definition展開を使い、XXPlusYY等の既存位相規約を維持する。
固定P0のRZ/SX/SXdg/X/CXだけは型とcontrol条件を確認して直接渡す。
Aerのqubit truncationと出力成分のzero thresholdは無効化した。
シミュレーション用変換はnative費用を数える回路を変更しない。

`c1_h4_native_fair.decode()` は新評価器を既定とし、独立照合用に
`backend='operator'` も残す。raw/compiled/actual raw inverse/actual compiled inverseを
それぞれ完全な512×512行列へ評価する。実際の逆回路をforward行列の共役転置で代用しない。
raw/reference、compiled/reference、compiled/raw、unitarity、inverse/reference、inverse return、
X/Yの既存判定式と許容差は変更していない。全入力・全出力を保持し、sector投影・少数probe・phase fitは行わない。

`c_pr_aware_df_stage_b2.trajectory_reference()` では、既存のhead/atom cacheに加え、
`event_unitary()` 自体の結果をcandidate-bound contextへ保存する。
keyはrotation component、rotation angle、phase、ordered product component IDsの全演算引数。
atom行列と保存event行列はread-onlyで、別candidateのJ/Uや別rの角度へ流用しない。
同時missは完成行列だけを公開し、重複計算が起きても部分行列を共有しない。

同J/U強Dの登録ではFutureで同じinput/candidateだけを共有する。
従来はD評価全体がregistry lock内にあり、無関係な候補も直列化されていた。
計算をlock外へ移し、独立Dを並列化した。失敗は依存先へ伝播し、retryは追加していない。

## 人工回路での測定

本番と同じCDF8のbuilder、relative Gaussian、P0 compilerを、固定seedから作った架空の
signed J/orthogonal U/one-bodyに適用した。保存実H4配列は読み込んでいない。
rごとに独立process、BLAS/OpenMP/Aer内部threadは1で測定した。

| r | 旧4行列の合計秒 | 新4行列の合計秒 | 速度比 | P0 compile秒 | 新旧最大成分差 |
|---|---:|---:|---:|---:|---:|
| 16 | 48.84 | 7.82 | 6.25 | 2.85 | 1.38e-13 |
| 32 | 67.56 | 11.47 | 5.89 | 4.79 | 2.09e-13 |
| 64 | 102.11 | 18.30 | 5.58 | 8.86 | 3.42e-13 |

「4行列」はraw/compiled/その実際の逆回路2個。人工測定のpeak RSSは約358〜399 MiB。
component許容差1e-10を維持し、新旧行列のbitwise一致は主張しない。
これらは実H4の実行時間、全パイプラインの速度比、科学的性能を保証する値ではない。

追加の人工P0でAer融合幅2〜6も測定したが、有意な時間改善は見られず、既定の融合政策を採用した。
人工・mockのeval回帰129件、fit回帰15件がPASS。
最後にD並列化/重複排除の追加2ケースを含む接続11件がPASSした。11件は前の検査と重複する。
位相差、sector外だけの誤り、誤った実逆回路、nested global phase、terminal measurement、
nonunitary拒否、full512 P0、event cacheと独立旧参照の一致を含む。

## このサーバーの構成

GPUA100-gleap02、EPYC 7532、affinity 64 CPU。
測定時MemAvailableは約492 GiB、保存先の空きは約153 GiB、loadは低く、
session/userのcgroup CPU・memory.maxはmaxだった。
A100-PCIE-40GB×4はGPU使用率0%。ただし固定qiskit-aer 0.15.1のavailable_devicesはCPUのみ。
今回はpackage pinsを維持するCPU実装を採用する。
squeue/qstat/bjobsおよびquotaコマンドは見つからず、未知のdisk quotaまで無制限とは認定していない。
共有設定・driver・他ユーザーのprocessやファイルは変更していない。

fit/evalは専用 `.venv-fit` / `.venv-eval` に分離した。
契約の主要17package pins、ffsimの指定2 SOURCE hash、extended accumulationを照合済み。
Pythonはサーバーの3.12.3で、元lockの生成注記3.11との差を記録する。
全package版は技術成果物の `fit-freeze.txt` / `eval-freeze.txt` に保存した。

既存の候補process並列とaffinity配分を再利用する。現在の資源では、開発fitは18process、
開発下流は最大36候補process、移送下流は最大12候補processになる。
各候補内のnative/round workerはそのaffinityに合わせる。BLAS/OpenMPは自job内で1、
Aer内部も1とし、候補processとnative taskでCPUを使う。
Qiskit build/compile/adjoint buildのprocess-local mutexは維持する。
これらはtask数と実測資源からの構成であり、過去の32 CPU/18並列を恒久上限にしない。

## 入力と新実行の境界

元開発input.npz、initial.npz、objective.jsonは登録hashと一致した。
旧execution SOURCEに列挙されたファイルは全件存在し、共通shell原本も利用できる。
ユーザーの指示により、ローカル停止時の18trajectory結果・最新STOP等は未取得として扱う。
したがって、新しい実行には元C0と固定2摂動のrestore、開発18trajectoryの再計算が必要になる。
これは旧runのresumeや、保存結果の存在を仮定した継ぎ足しではない。

H4/1.15 Å入力は登録されたgeneratorから、開発recipeの凍結後にだけ生成する。
開発の数値J/U、ground、価格を移送入力に代用しない。
共通r32と公平r16/32/64、selection32と独立final64、同J/U強D、13round、
`W * mean_kernel_cost + shell_cost`、shots/W/kernel/shellの分離、全科学閾値、
最大172,835 unitsとnative最大7,680件は維持する。

差分は新SOURCE、interpreter path、完全行列評価器、event cache、D並列化、
新出力配置と技術的な計時。旧contract/SOURCE/raw/STOPは変更していない。
新SOURCE digestが乱数namespaceへ入るため、新しい独立cohortであり、旧標本の再現を主張しない。
科学的な標本数・候補数・検査範囲を減らす契約変更は提案していない。

## 次の実行

新出力は `artifacts/research_c_integrated/2026-11-06/{development,transfer}/run01`。
どちらも未作成。新familyのSTOPも未作成。retry/resumeは0。
passive確認は次のコマンドで行える。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-integrated-server-20261106
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-eval/bin/python -B scripts/prepare_research_c_integrated_server.py
```

**ユーザーが上記新SOURCE/契約、欠けている開発18trajectoryの再計算を含むfresh科学実行を明示許可した後にだけ**、
同コマンドへ `--execute` を付けてCodexが開始する。
ユーザーの引継ぎ指示にある「変更した実装を科学計算に投入する場合は、新しいSOURCE・固定契約・明示的な実行承認が必要」に従う。
承認証明書や承認原文の保存は要求しない。

最初の登録native32のwall/CPU/peak RSS・affinity/worker計画を `native_timing.json` に保存し、
統合進捗へ実測と最大native作業量の粗い外挿を出力する。
Codexはその値と、既に実測した初期化/最適化/PF等の時間から所要時間を更新する。
人工測定だけから実H4の完了時刻を確定しない。標本数の事後追加・outlier除外は行わない。
