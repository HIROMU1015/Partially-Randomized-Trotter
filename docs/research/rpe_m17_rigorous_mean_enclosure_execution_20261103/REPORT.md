# RPE m17：指定ideal categorical scopeの十分条件を包含で確認

実行2026-11-03 23:58 JST完了、報告・公開作業は2026-11-04 JST。
**固定SOURCEを1回実行し、96 bitの1包含でcosine/sine両軸のstrict十分条件を確認した。**
結果は`CONDITIONAL_IDEAL_CATEGORICAL_SUFFICIENT_CONDITION_CONFIRMED`。
これは、今回SOURCE replayした成分/basis、IID uniform53抽選、ideal未compile gate、保存stateのnormalized rayに限定した結果。
実PRNG列・float仮想測定・compiled回路/noise・実state preparation等へのbridgeはnullで、
全体の`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`を維持する。
旧数値診断INCONSISTENTや旧u_num_bound=nullを遡及変更しない。

## SOURCE・固定物理条件

科学実行SOURCE：`18fffce48e33136b91eb6ac5a6f9a5f77a08d36f`。
[固定計画](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)、
[保証scope](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/ERROR_BOUND_SCOPE.md)、
[adapter/人工49 PASS](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/IMPLEMENTATION_REPORT.md)を変更せず実行した。
実行branch：`codex/rpe-m17-rigorous-mean-enclosure-execution-20261103`。
科学SOURCEと今回の結果公開commitは役割を区別する。

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bのphysical ground-stateベクトルのray、
8qubit・4電子sector dimension70、Partial L_D3、単一δ=.02、m17/q131072/r32/K2。
extract_identity_phase、coefficient_atol=1e-12、各軸N411、保存α=.016957089147912167、
保存ε=.15244325851620388、保存PF/RTE/stat予算を保持した。
μ0はV2保存model平均の同じbinary64値（Re .32902182375994465、Im −.46875529425442597）。
α/εは元m17の残余評価と同じ保存decimal文字列のexact rationalとして解釈した。

入力は元m17結果 `cd749066904d3f29b28b03c19ab69370372b157f`のRESULT/7行列、
B state `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`、Task A結果 `a5304864ec5e1412a36c31574bcb95fb13f80202`のRESULT/S64、
B-1a結果 `f766a1f7fc0cd9131a21d0594da242865fe1cb0d`のRESULT、
carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の保存H snapshot。
7入力のpath/SHA256は固定計画と今回の原RESULTに記録。
元operators/S64はbyte/header保全確認のみで、新しいideal演算子として流用していない。

## 再生成・categorical law・ideal測定の定義

component/basis preparation/export1回で218成分・11 basis・4 deterministic blockを取得。
preparation hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、
tail hash `65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af`、保存λ_R=.5832704246339503の照合を通過した。
prefixはDF index[0,1,2]、tailは[3,4,5,6,7,8,9,10,11]。
成分順・係数/sign/probability、basis descriptor/local matrix、primitives/phase等をexportした。
これは`SOURCE_REPLAY_GENERATED`であり、過去の未保存payload原本を取得したものではない。

componentとorderのCDFを各1回構成し、NumPy1.26.4のcumsum/最終値による正規化を再現。
固定CDF64境界に対し、独立な一様53-bit wordを仮定したbucket massをexact rationalで保存した。
RNG replay、PCG64 sampling、量子shot、標本頻度からの推定は行っていない。
componentの再正規化・choice内部CDF・finite-grid massを区別し、元event_probabilityのmetadataを実装確率として代用していない。

理想basisはbinary64パラメータをexact realとしたRZ/XXPlusYYの解析式で包含した。
これらとZ/ZZのexact粒子数保存を使い、完全sectorへ作用させた。数値leakage=0.0を厳密な根拠にしない。
θ64はB-1aで新生成して保存された値を固定し、π/B/θの定義式からの新生成やB-2は行っていない。
signed component、event phase、F→r event→R、q反復、cosine=Re/sine=+Imを保持する。
確率lawにattenuationが含まれるため、追加のB除算はしていない。

保存stateのraw/fingerprintを変えず、ideal量子測定の入力をnormalized rayと定義した。
期待値をψ†S^qψ/(ψ†ψ)で包含し、norm²区間を原RESULTへ残した。
raw stateのvdotを使う過去のfloat仮想信号とは区別し、実state preparationや真の分子基底状態への距離を認定しない。

## 包含結果と固定予算

原記録：[RESULT.json](../../../artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/RESULT.json)。
以下は原RESULTのexact rational endpointを読みやすく表示したもの。
表のν_cat区間は小数で外側へ丸めている。正式なendpoint/U_cat/sとstrict判定は原RESULTのrational値に従う。
報告作成時にoperator/mean/norm/CDF/予算を再評価していない。

| 軸 | ν_catの包含区間（外側に表示丸め） | U_cat上端（概数） | categorical判定 |
|---|---|---:|---|
| cosine | [0.32902140102557, 0.32902224674832] | 4.2298837484e-7 | 十分条件確認 |
| sine | [−0.46875571719094, −0.46875487146819] | 4.2293650642e-7 | 十分条件確認 |

両軸のsは約.1523572519941944950289、ε=.15244325851620388。
固定SOURCEの外向き比較により、両軸で`upper(U_cat+s)<lower(ε)`が成立した。
U_catはこの指定ν_catと保存μ0の座標差を覆う上端であり、Law/referenceの観測差とは異なる。
今回の成立は局所の保守的十分条件についての結果で、RPEそのものや全roundの成功確率を新評価したものではない。

### 誤差半径とphase transfer

| 内部項目 | 原RESULTの表示（概数） | 意味 |
|---|---:|---|
| logical signed平均 Re | [0.3290218238869495754417, 0.3290218238869495754891] | exact ±1 event phaseを使う補助平均の包含。表示は概数 |
| 同Im | [−0.4687552943295624647815, −0.4687552943295624647342] | 同上 |
| powered operatorのglobal radius上端 | 2.3666770562e-20 | Acb/global norm伝播で得た数値包含の半径 |
| builder π64 phase transfer上端 | 4.2286136912e-7 | π64と厳密πとの差・sequence加算丸め・q反復を保守的に転送 |

ν_catにはbuilderのπ64加算を含めるため、logical平均の区間だけで判定せず、phase transferを両座標へ加えた。
これは実行時に位相差を観測・fitした値ではなく、固定計画の一様上界。
半径や高精度表示桁を実測定全体への保証と読み替えない。
96 bitで条件が確定したため、計画どおり160/224 bitは未実行。precision間収束を保証根拠にはしていない。

## 未保証bridgeと判定の限界

次のbridgeは原RESULTで全てnullのままである。

- 固定PCG64列からIID一様抽選モデルへの妥当性。
- float state-action・clipping・Bernoulli比較を含む仮想測定平均との差。
- controlled-gate内部合成・compiled回路、noiseを含む実機。
- 実state preparation、および過去の未保存component payloadとの完全なidentity。

そのため各軸のU_exec_upper=null、exec_decision=UNDETERMINED。
categorical条件確認を全体のRPE_tolerance_certifiedへ昇格させず、scientific_status=UNDETERMINEDを保持する。
旧INCONSISTENT/旧u_num=null、V1/V2/V3/R1・候補A/Bの結果を変更していない。
PF/RTE/stat予算、N/α/β/εも変更・二重加算していない。

## 実行・資源・原本

call数はpreparation/export1、component CDF1、order CDF1、complex enclosure1（96 bitのみ）。
中断・retry・追加精度は0。SCF/DF因子化/H生成/state solve、sampling/shot、controlled回路build/compile、
費用・成功確率再解析、独立詳細科学レビューは0。
実行wall time約3.717秒、うち包含call約2.397秒、exit0でCOMPLETED_MANDATORY_STOP。

起動時CPU affinity32、load average約1.60、利用可能RAM約51 GiB。GPU driverは利用不可。
python-flint0.9.0/FLINT3.6.0、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、qiskit-nature0.7.2、
OpenFermion1.6.1、mpmath1.3.0。FLINT threads32、BLAS/OpenMP/MKL各4（自分のprocessのみ）。
component平均のbasis共有とbinary powerを使用し、event全列挙・trajectory生成を避けた。
process RSS/実CPU占有率/serial速度比較は測定しておらず、速度向上率は主張しない。

```bash
PYTHONWARNINGS=ignore MPLCONFIGDIR=<owned-temp-dir> OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/run_mean_enclosure.py --run --output-dir artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01
```

| 保存原本 | Bytes | SHA256 |
|---|---:|---|
| RESULT.json | 7695 | `b58caa9dd8a0267eb20c049636e6d8c0b56d67125912e6b5a6a872334a1a2938` |
| source_replay.json | 763344 | `ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08` |
| categorical_laws.json | 29314 | `a75324a1ca014bedf56fbd4e8eacbe6c9829f4bfbc3f8ccf248b5db6cc897e91` |
| enclosure_operators.zip | 1266687 | `708095a92f7d3080f595388b80be6b6b9a26b4e4536f416edd8d7434731a5d56` |

ZIPは96.jsonの1memberで、中間operator7個のexact center/global spectral radius/norm上端を保存。
byte hash・ZIP整合性・schema・原SOURCE/input保全を確認した。これは独立詳細科学レビューではない。
他branch/worktree・SERVER2 H8・共有環境を変更せず、専用結果branchにだけ記録した。

次段案：**ユーザー**がどのν_exec（仮想測定、compiled回路、実機等）へ接続するかを選ぶ。
**Codex**はその指定後に対応bridgeのSOURCE・既存証拠・不足を静的に整理し、必要な限定検証を提案する。
新しい科学実行・他round・B-1b/B-2・sampling/compileへ自動移行せず、ここでmandatory STOPする。
