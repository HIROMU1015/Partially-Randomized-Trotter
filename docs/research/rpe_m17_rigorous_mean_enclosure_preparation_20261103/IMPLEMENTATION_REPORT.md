# m17測定平均の包含adapter・SOURCE準備

2026-11-03 JST。**export、CDF/grid、ideal gate、Acb/global radius、固定実行runnerを実装し、人工入力で検証した。**
実H4のcomponent/basis再生成、CDF・bucket mass、数値operator、norm、包含mean、予算再評価は行っていない。
[保証scope](ERROR_BOUND_SCOPE.md)、[固定科学実行計画](SCIENTIFIC_EXECUTION_PLAN.md)、
[次段判断](NEXT_STAGE_DECISION.md)と合わせて読む。

## 再利用と新実装

| 既存資産 | 再利用方法・今回必要な差分 |
|---|---|
| carrier4591b8d…のsnapshot loader、split/prepare_df_partial_s2、tail extraction/normalization、basis registry | export_adapter.replayが将来の--run内で一度だけ呼ぶ。生成済みsnapshotを使い、SCF/DF因子化/state solveは呼ばない |
| 元m17 SOURCE128e9a5… / 候補A修正SOURCEd0b1da1… | load_numerical_api、source_and_input_identity、sector permutation、固定hashを再利用。prepare_context全体はmean生成まで行うので流用しない |
| df_trotter.opsのdiagonal primitives | halfのbinary64 primitive入力をそのままexport。新たなPF係数生成実装は作らない |
| Task A/B-1aのbyte/header照合とatomic RESULT writer | 元入力・SOURCE保全確認に再利用。旧float/HP meanを計算し直さない |
| h4_ball_evidence.py / h4_arb_bridge.py | rational→ball、exact endpoint、Frobenius上端、global radiusへ移す純粋な数値式を限定的に再実装。旧研究のentrypoint/guard/grantを変更・転用しない |
| 元finite-RTEの平均式・sampler・Hadamard convention | Lawを保持し、CDF/gridを明示。Taylorへ置き換えて数値差を消す変更はしない |

既存NumPy/mpmath adapterは包含radiusを保持しないため、新しいbackendが必要だった。
既存Arbの生産entrypointは別研究の固定scopeを持つため、新adapterを分離した。
新規管理基盤や承認証明機構は作っていない。

## 新ファイルの役割

- [export_adapter.py](export_adapter.py)：成分ID/順序・係数/sign/p64/λ、basis descriptor/local matrix、
  primitives、prefix/tail、preparation/tail hash、state/ordered sector、独立scalar角をexport。
  完全な原payloadではなくSOURCE_REPLAY_GENERATEDと記録する。標準RZ/XXPlusYY以外はSTOP。
- [mean_adapter.py](mean_adapter.py)：SOURCEと同じcomponent再正規化・choice CDF、exact 53-bit grid mass、
  number-preserving ideal basis、signed component平均のbasis集約、event/r/F/R/q、builder phase transferを実装。
- [ball_adapter.py](ball_adapter.py)：exact rational/dyadic入力、Acb演算とglobal spectral radius、
  binary power、正規化ray期待値、外向きendpoint、strictな十分条件・null bridge処理。
- [run_mean_enclosure.py](run_mean_enclosure.py)：defaultはbyte/JSON/headerのみ、--runで別許可の科学処理。
  最大96/160/224 bitの3包含、fresh出力、原本・部分失敗・STOPを保持する。

実H4のデフォルト動作をnp.load/replay/CDFのmock禁止下で検査した。
--runは操作者が許可範囲を守るための明示選択で、承認を認証する仕組みではない。

## SOURCEの意味論で固定した点

Gaussian basis gateは登録されたbinary64パラメータのexact ideal gateとして定義する。
exportするfloat local matrix/hashはSOURCE対応用で、float matrixを厳密unitaryとは扱わない。
RZ/XXPlusYYとZ/ZZの解析式からexact sector不変性を使い、元leakage=0.0を証明として転用しない。
gateのlittle-endian wire順とsector permutationは人工Qiskit referenceへ照合した。

回転のnegative signはsigned involutionに入り、eventの(−1)^(n/2)とは別である。
controlled builderのRZ/RZZの倍角が有限・exact doublingであることを将来runで確認する。
F/Rのscalarはcontrolled SOURCEが別々に適用するconstant/identity角を保持。
RTE sequenceのπ64加算もlogical ±1との差を外向き上界に入れる。詳細はscope文書。
この差を観測gapの原因と特定した結果ではなく、包含で必要となる静的SOURCE対応の確認である。

保存ψは書き換えず、ideal測定入力をそのnormalized rayと明示する。
float vdot・clipping・Bernoulli比較との差、PRNGのIID性、過去payload・compiled/noise・state preparationのbridgeはnull。
categorical十分条件だけを確認できても、RPE全体の科学statusはUNDETERMINEDのままである。

## 環境と再現

本人所有の独立venvへpython-flint0.9.0を追加し、FLINT3.6.0を確認。
既存研究のnumpy/scipy/qiskit等はread-onlyで参照し、元venvへinstall・変更していない。
実行時に確認するversionsは[requirements.txt](requirements.txt)の7項目。
公開文書には個人の絶対環境pathを含めない。再作成は本人の分離venvで以下を使える。

```bash
python -m venv <dedicated-venv>
<dedicated-venv>/bin/python -m pip install -r docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/requirements.txt
```

wheel内部のFLINT versionもrunnerが確認する。共有package/system-wide環境を変更しない。
小さなテストのFLINT/BLASはthread1、科学runnerはaffinityからFLINT threadsを選ぶ。
逐次matrix chainはC backend内部を並列化し、event全列挙や同じmeanの重複実行を避ける。

## 人工検証・方法選択

[test_mean_enclosure.py](test_mean_enclosure.py)は人工RZ/XXPlusYY・2次元sector・rational行列・CDFだけで検証。
変換、包含endpoint、signed非可換平均、r/q・作用順、normalization一回、state ray、
Re/+ImとSdg/HのHadamard、π64 phase上界、unknown bridge、strict ε、非有限/不正入力、
fresh出力・中断記録、precision切替、default非科学動作を確認した。
export_preparationは人工preparation objectで検証し、実H4へ呼んでいない。

人工8×8 dense unitary、q131072、96 bitの比較では、global spectral radius上端は約1.1492e-22、
直接entrywise powerの[0,0]実部radiusは約1.9454e-17だった。
両者は異なる種類のradiusであり速度比・実H4の精度見積りではないが、依存性の増幅を抑えるためglobal方式を選定した。
224 bitの独立な人工参照を含むことも確認した。
実H4の幅・runtimeは未測定で、判定力を保証しない。

新規25件＋元m17人工13件＋既存静的人工11件＝計49件PASS。
環境・実行記録は[VALIDATION.txt](VALIDATION.txt)。LOCALの技術試験で、独立科学レビューやCI証拠ではない。
テスト初期にtoyのQiskit convenience API名・Fraction変換・alphaのbinary64対rationalの混同を直し、最終suiteを通した。
scope/閾値/実H4入力を変更してPASSさせたものではない。

## 保全と到達点

元m17/Task A/B-1a、V1/V2/V3/R1、候補A/BのSOURCE・raw・status・STOPを変更しない。
新しい科学artifact folderは未作成。research synthesis/validation manifestの科学結果も変更しない。
既存dirty/untrackedや他worktree、SERVER2 H8、共有環境に干渉していない。
準備SOURCEのcommit/push・公開照合後にmandatory STOP。実行には固定計画への別の明示許可が必要。
