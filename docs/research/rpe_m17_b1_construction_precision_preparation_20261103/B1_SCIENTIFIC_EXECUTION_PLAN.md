# 第一科学検証B-1aの固定計画（未実行）

2026-11-03 JST。この文書の作成・公開は実行許可ではない。
本フォルダを含む公開commitをB-1a実行SOURCEとして固定し、別途ユーザーが実行を明示許可した後だけ実行する。
入力の細目・数式・SOURCE境界は[B1_INPUT_BOUNDARIES.md](B1_INPUT_BOUNDARIES.md)。

## 科学的な問いと固定条件

保存tail以前の構成算術を高精度化すると、Task Aに残ったLaw/reference差がどの程度変化するか。
比較基準とB-1aの後段は同じHP実装・同じF/R/ψ/qにして、outer精度変更とtail変更を混ぜない。
構成算術への依存を検出できるが、保存前の表現誤差・scalarとmatrixの個別寄与・定義差の一意な識別までは主張しない。

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bのphysical ground state、
4電子sector70、Partial L_D3、δ=.02のみ、m17/q131072/r32/K2。
`extract_identity_phase`、`coefficient_atol=1e-12`、保存N/α/ε/βを保持。
SCF/DF/Hamiltonian/state/basis/成分/新F/Rの生成、sampling、compile、費用・energy再解析、別roundは0。

## 固定input

以下はリポジトリ相対path。prefixは省略表記。

| Commit / prefix | File | SHA256 |
|---|---|---|
| cd749066… / `artifacts/rpe_m17_mean_validation/2026-11-03/run01/` | RESULT.json | `79fdf7e84f90c48230857b13f39f9becb30f19892e2e9c09d80805d346432a4a` |
| 同上 | operators.npz | `e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df` |
| e22d7718… / `artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/` | physical_state.npz | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| a5304864… / `artifacts/rpe_m17_precision_isolation/2026-11-03/run01/` | RESULT.json | `e5c3cb1cd455d102c8a1e74d7556ec95b26ab79e72f0727e172f4483162f362e` |
| 同上 | frozen_steps.npz | `2250e10bcae320c2fe1ec30662eb9ce6e14df50798f3db2d78f60c28dcfd7c30` |

実際のfull commitは入力境界文書とrunnerの定数で固定。原本とgit blobのbyteを照合する。
frozen_stepsは保全確認のみ。Sの数値再評価・旧Task Aのmeanの再計算には使わない。
Task A `fixed_factors.best_pair`と収束記録を文字通り再利用する。

## SOURCE・順序・call範囲

[driver](run_b1_construction.py)、[adapter](b1_adapter.py)とTask A/元m17の既存helperを使用。
再利用元を固定SOURCEとbyte照合する。新2ファイルは実行HEADと一致させ、SOURCE commit/hashを数値再現性のために記録する。
承認の証明書・ID・grantは作らない。

1. 新規出力directoryだけ作り、元入力をread-onlyでロードする。
2. 保存τ/π/BとSOURCE replay θ64等を区別して記録。旧float LawとTaylor関数でevent/polynomial/tailを各一度構成する。
3. 保存event/tailとの差、array_equal、Frobenius差を記録する。bit一致を前提にしない。
4. float replay tailが保存tailと同じ経路は、Task A保存factor固定meanをそのまま使用する。
   異なる経路だけ新replay tailを固定したHP後段評価を行い、元Task Aとの差を明記する。
   非一致を理由に旧SOURCEを変更したり、byte一致まで反復したりしない。
5. 保存A_prob/A_coeff・π64・τ64・B64・新θ64を正確にlift。
   HP cos/sin、event和、Taylor除算/積/和、内側r乗、reference normalizationを評価する。
   得たtailをcomplex128へ落とさず、Task Aと同じHP `(R*tail)*F`、q乗、期待値へ渡す。
6. 収束した新tail/両mean、Task Aとの差、新replay基準との差、Law/reference差を保存して終了する。

float SOURCE replayはLaw event/tail各1とreference polynomial/tail各1。
B-1aは各precision・各pathにつきtail構成とmeanを各1回。
初期96/160 bit×2path＝4 mean、必要な場合だけ224 bit×2pathを追加し最大6 mean。
replay差がある経路だけ基準meanを各2、必要なら各1追加（最大6）。合計HP mean最大12。
同じ旧tailのmeanは再計算しない。中断時の部分完了は記録し、最大call数を増やして成功を狙う反復はしない。

## 精度・収束・判定

mpmath1.3.0 CPU多倍長、binary powering。固定matrix/ψ/scalarをdyadicとして持ち上げる。
96/160 bitの両path mean差が各1e-13以内、かつtailのFrobenius変化が各`1e-13/q`以内を収束診断とする。
tail基準はqによる増幅を意識した技術的診断で、行列normや安定性を含む厳密摂動上界ではない。
不収束時だけ160/224 bitで再確認。224 bitでも不収束なら追加精度なしでSTOP。
conditional基準が不収束ならB-1aを開始せずSTOPする。
Task Aの96/160収束をB-1aへ無条件に転用しない。scalar比較contextは288 bit。

| 判定 | 規則・解釈 |
|---|---|
| 数値収束 | meanとtailの精度間変化が事前基準内。真値誤差の保証ではない |
| 構成の数値診断 | 収束後のcomplex mean gapを元1e-10と比較。±1e-13の診断band内はNEAR_THRESHOLD |
| BELOW | gap < 1e-10−1e-13。構成算術への依存を支持。単一原因や真の平均との一致は証明しない |
| ABOVE | gap > 1e-10+1e-13。固定A/π/θ/Bの表現・定義差等が残り得る。実装bugと断定しない |
| 不収束・欠測・非有限 | INSUFFICIENT_EVIDENCEまたはINTERRUPTED_OR_INVALID、mandatory STOP |
| RPE適合性 | u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを保持 |

mean差はRe/Imの符号付き差・絶対complex差で報告。tail Frobenius差を最終gapの寄与率にしない。
Task AからB-1aへの変化は、replay driftと高精度構成変更の二段を区別する。
新旧の後段は少なくとも160 bitの収束した同一アルゴリズムで比較し、各結果の使用bitを記録する。
元1e-10は診断用で維持するが、RPE判断にそれ以下のgapを必須としない。
保存統計残余約8.60e-5は元結果の引用。真の周辺平均への誤差を残余内で評価できる保証が別に必要である。

## 資源・出力・終了

Task Aでは70×70のHP後段が主な負荷。独立なpath×precisionの4jobをspawn並列し、
CPU affinityと実際の空き資源に合わせる（4は独立job数でありCPUの人工上限ではない）。
float replayのBLASとprocessの多重化を調整する。実行前に空きCPU/RAM/GPUをread-onlyで確認し、
他jobの設定を変更しない。wall-time/CPU/RAMの人工capは設けない。
人工行列での実spawn試験を事前実施。実H4の新timing/pilotは準備中に実施しない。

将来の出力先：`artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/`（今回は未作成）。

- RESULT.json：SOURCE/input、scalar provenance/hex、replay差、使った旧基準、新mean/残差、収束/判定、資源・wall time、完了/中断状況。
- float_source_replay.npz：新float event/polynomial/tail。旧保存配列とは別の原本。
- high_precision_operators.zip：mode/path/bit別にHP one-eventまたはpolynomialとtailをdecimal round-trip JSON保存。float64へ丸めない。

実行コマンドは別承認後だけ使用する。

```bash
python docs/research/rpe_m17_b1_construction_precision_preparation_20261103/run_b1_construction.py --run --output-dir artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01
```

出力の既存directoryへ上書きしない。exception/interruptでは部分RESULTと回収済みoperatorを保全し、workerを回収してSTOP。
過去の技術中断修正・再実行許可を使う場合も、失敗原本を保全し、技術修正版SOURCEを固定し、科学条件・call/精度範囲を変えない。
正常なABOVE/NEAR_THRESHOLD/不収束を技術中断として繰り返してはならない。

終了後は旧m17/Task A/V1/V2/V3/R1/A/BのSOURCE・raw・STOPを維持し、新結果だけ報告してmandatory STOP。
B-1bは成分の固定入力が揃い、A構成の追加分離に意味がある場合だけ別提案。
B-2は整合的な定義式入力の再生成が必要な場合の別候補。どちらもこの実行範囲に含めない。
独立詳細科学レビュー、他round、R2/R3、追加samplingも自動開始しない。
