# H4 基底inverse・命令/定義encoding再利用：稼働runを維持した軽量実装

SOURCE `def7d48f4af6c07f2d4e361fae868f3b01db8915`、branch `track-a-h4-inverse-instruction-speedup-20261011`。
起点REVIEW `d0b4e1734b063c9fef353fdb3a7b63b26c961e72`、稼働SOURCE `941749b8d4aedb43fd5f47b824116082493df393`。
[固定source・限定検査・軽量review・未採用binding](../../artifacts/resource_applicability/track_a_h4_inverse_instruction_speedup/2026-10-11/README.md)を資料入口とする。

## 変更と範囲

利用者の「１，２を行って、現在のrun は止めなくてよい。レビューも簡単でよい」に従い、独立worktreeだけを修正した。
[GeometryPreparation](../../src/trottertracks/resource_applicability/h4_geometry/signal.py)がgeometryごとの既存基底と一緒にinverseを一度保持し、
[build_evolution](../../src/trottertracks/resource_applicability/h4_geometry/circuits.py)が同じbasis transitionで再利用する。
基底とinverseはgeometry準備内で所有し、builderはcomposeでコピーする。未cache blockには元のinverse呼出を維持する。
変換の順序、数値、phase、controlled operation、Gaussian gate sequenceを変えない。

[InstructionEncodingMemo](../../src/trottertracks/resource_applicability/h4_geometry/instruction_encoding.py)は一つのnumerical fingerprint内だけで、
小さい命令と定義の旧canonical textを再利用する。各使用時に現在の値からkeyを再構築するため、mutable object idに依存しない。
parameter、dtype/shape、signed zero、qubit/clbit位置、condition、control state、label、定義のaxis/global phase/順序を含む。
cacheは保守的に計上した2MiB・2048 entries以内のLRU、key走査は16KiB・384 nodes以内。
定義は16命令以下で子命令が既に小さいEncodedValueの場合だけfragment化する。
大きなarray/定義、cache不適合値は元のLazyList走査に戻し、全回路のexact treeを作らない。65536-byte chunkと独立observerは保持する。
公開serialize/deserializeの既定形式を保持し、fingerprint経路でだけmemoを有効にする。

source closure89：旧85中2変更/83不変、新module・runner・tests・親oracleの4追加。
sampling/seed/signal関数はAST一致、compiler/options/worker/IPC/observer/budget/binding sourceはbyte一致。
H4 linear/STO-3G/DF rank12・6 geometry・218 template・32 paired trajectoriesの既承認signal/compile map、wrapper semantics
`h4-full-gaussian-givens-paired-wrapper-v2`は維持する。17GiB/74805 actual/累積72h、worker4/32GiB、driver8GiB、observer256MiB/64MiB、内部thread1も維持する。

## 限定検証と簡単なreview

[実行前計画](../../artifacts/resource_applicability/track_a_h4_inverse_instruction_speedup/2026-10-11/test_plan_v1.json)は単一test process、AS1GiB/RSS512MiB、wall60秒、output4MiB。
既存private venvを変更せずPython -P -B、process-local thread1/Qiskit num_processes1。child/affinity/transpile/GPU/実NPZアクセスは禁止する人工検査runnerで実行した。
14件PASS、failure/error/skip0。wall 1.635秒、kernel peak RSS 99614720 bytes、最終証跡 4294 bytes。
親resume01 serializerと旧run05 oracleに対してfull canonical bytes/digest一致。
paired axis、open/closed control、phase、label/condition、順序、definition/matrix変更、nonfinite/symbolic/recursion STOP、LRU上限、大きい値のlazy fallback、公開deserializeを検証した。
回路のgate counts/depthは人工uncompiled fixtureで一致。compiled metricsは追加transpileを行わず未検証とする。

一つの人工buildでinverse再計算9回→0回。
一つの640-instruction人工fixtureの直列化は親 0.445597秒→新 0.319781秒、約1.39倍。
追加campaign/人工compileは0で、production全体の高速化率・完走時間の実測値ではない。
[軽量review](../../artifacts/resource_applicability/track_a_h4_inverse_instruction_speedup/2026-10-11/delta_review_v1.json)は変更差分と不変条件だけを実装authorが確認し、blocking findingなし。
独立最終launch reviewや旧campaignを繰り返していない。local evidenceでありCIや最終科学結論とは扱わない。

## 現在のrunと将来の採用

`h4-newhost-signal-compile-20261011-resume01`のworktreeとsource85件のSHAを読み取り照合し、一致を確認した。
今回の実装は稼働runへ反映しておらず、runは既定のobserver下で継続する。
新準備bindingはapproved=false/runtime_authorization=false/allowed_cpus=[]/unsealed/absolute_launch_command=null。
新SOURCEの将来の採用にはsource/profile/seed/input/carry/plan/auth/review再結合とfresh runが必要。
現在のcontinuationはrun10→resume01の一度の移行専用であり、resume01からの次の引継ぎ処理はこの修正の範囲外。
元output/control・旧証拠・消費budget・seedは今回変更しない。新しい本体起動は行っていない。

## 公開

SOURCEはsource/testsの6fileだけを固定。sourceを変えない別REVIEW_BUNDLE commitに限定結果・監査・軽量review・binding・関連indexを固定する。
NPZ/実runtime/checkpoint/cache/log/認証情報をcommitしない。originへのnon-force pushに認証が使えない場合は設定を変更せず、次の手動commandを使う。

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-inverse-instruction-speedup-20261011 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-inverse-instruction-speedup-20261011
```
