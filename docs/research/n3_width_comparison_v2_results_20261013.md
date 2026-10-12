# N3幅比較v2：固定6入力pilotの一次結果

2026-10-13 JST。[明示認可](n3_width_comparison_v2_execution_authorization_20261013.md)に基づく一回の固定pilotを完了した。
**全6入力・504 mode-method recordsが完了し、独立保存監査PASS。目標幅.01ではNORM/OCC/同値SDP/SOSが全6Hを縮約でき、OCCの追加H縮約は0。**
有限保存幅と取得費用には差がある。中心テーマ・新規性・継続GOはCodexで採択せず、GPTの科学的判断を待つ。
statusは`FINITE_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_REVIEW`、認可消費1、retry0、Mandatory STOP。

## 固定した条件・provenance

3stress family×full modes5/6の6Hamiltonian、ν2/3、各Hの候補{0,n−1}、一外部mode、実対称h、非負density相互作用、ε_proj=1/100。
化学geometry/basis、DF rank/policy、L_D、PF時間窓は適用なし。係数recipeの共通energy単位を使う。
[全係数・seed](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json)と[結果前設計](n3_width_comparison_v2_preregistration_20261012.md)を維持した。
集合は事前開示したstress collectionであり、IID分布性能・旧2-base成功と扱わない。原12/8入力とS1を新成績へ加えていない。

source `3eba0814e690e95993db60809c4c3c96784b07c0`、認可commit `fcc75c9ff442df5368c316fd1cc16c475278f3c8`。
source fingerprint `9f7687e74fe93f4b238579ee70b846521f4ea296195491c9f22e536a60066d96`。
input SHA `4c59ba790c7a1faefa6e60dbfbde3d13b718e168bb6491fa71594103b3ac8395`、plan SHA `dc43a8d781e76ba4c3fc92f528c973f225861bf47160c5866bff2bebd9eb9d53`。
[launch receipt](../../artifacts/n3_width_comparison_v2/2026-10-12/run1/launch_receipt.json)はsource/input/plan/authを結合し、source13 blobも実行後一致した。
2spawn workers/thread1、AS3GiB/worker、aggregate sampled RSS6GiB、output128MiB、wall/CPU capなし。
環境・package pin・有限探索・solver3tiers/20000iterations・proposal40bits・checker/call/bit上限を変えていない。

## 目標幅.01での縮約判断

kernel/coldの各3反復で下記の採否は同じ。反復を独立入力として分母を増やさない。

| method | 一mode縮約を保証したH | 受理候補mode |
|---|---:|---:|
| NORM | 6/6 | 12/12 |
| FIVE | 5/6 | 7/12 |
| OLD | 5/6 | 8/12 |
| OCC | 6/6 | 12/12 |
| SDP_EQUIVALENT | 6/6 | 12/12 |
| DENSITY_SOS | 6/6 | 12/12 |
| FULL_H_LOWER | 0/6 | 0/12 |
| CHEAP_ORと各共通安全OR | 6/6 | 12/12 |

H単位の削減は最大1mode。追加の候補受理、元H fallback、安全ORの利益を追加qubit削減やOCC固有利得へ読み替えない。
FULL_H_LOWERは有効証拠を取得したが、目標幅には達していない。物理的縮約不能を示す結果ではない。

## 有限保存幅

各Hの2候補modeに対する有限保存certificate poolの最小幅上端。表示値は丸めており、厳密な有理区間は[保存集計JSON](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/summary.json)と[全比較区間](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/pairwise_saved_intervals.json)を参照。
理想保証族の最適幅、真の投影誤差、真の基底エネルギーを意味しない。

| input | ν | NORM | OCC | 同値SDP | 固定SOS |
|---|---:|---:|---:|---:|---:|
| V2_banded_5 | 2 | 0.0001930129718 | 0.0001930129718 | 0.0001932399029 | 0.0002032849353 |
| V2_banded_6 | 3 | 0.000583044542 | 0.0003225371181 | 0.0003225452826 | 0.00130977732 |
| V2_dense_density_5 | 2 | 0.002103748154 | 0.001232228197 | 0.001232291947 | 0.001275955414 |
| V2_dense_density_6 | 3 | 0.00159841369 | 0.001518114233 | 0.001518427394 | 0.001585533061 |
| V2_two_external_stress_5 | 2 | 0.002940394355 | 0.001988397586 | 0.001988453921 | 0.001745617339 |
| V2_two_external_stress_6 | 3 | 0.004820368414 | 0.004770665142 | 0.004770908586 | 0.004950524214 |

保存幅はkernel/cold間、同層3反復で一致した。非FULL対照の同一U,cを72比較cellでexactに確認した。
OCCはCHEAP_ORよりH最小幅で5/6、同mode幅で10/12がstrictに狭く、残りは同一区間。
OCCは同値SDPよりH幅6/6、同mode12/12でstrictに狭いが、**同じ理想条件を有限budgetで取得した証拠の差**である。
raw SOSとの同mode比較はOCCが狭い6件、SOSが狭い6件。H最小幅はOCCが狭い5件、SOSが狭い1件。
全法へ同じcheap ORを付けたcoverageも別保存した。raw SOSがOCCより広い例から、包含済みの理想SOS条件が弱いと結論しない。

## 取得費用

cold pathの3反復median、単位ms。係数bound取得、候補生成、有理化/修復、独立checker、幅、selector、in-memory serializationを含む。
workerの依存importと共有case-file disk IOは除外して別計上。全7法のkernel/cold median/range、CPU、phase、calls/bit/bytesは保存集計を参照。

| input | NORM ms | OCC ms | 同値SDP ms | 固定SOS ms |
|---|---:|---:|---:|---:|
| V2_banded_5 | 7.100 | 33.299 | 69.440 | 1195.192 |
| V2_banded_6 | 11.733 | 57.320 | 111.341 | 5776.770 |
| V2_dense_density_5 | 7.116 | 41.790 | 70.162 | 1603.426 |
| V2_dense_density_6 | 11.548 | 71.176 | 168.523 | 6500.116 |
| V2_two_external_stress_5 | 7.104 | 42.275 | 67.851 | 1211.030 |
| V2_two_external_stress_6 | 11.449 | 71.104 | 124.597 | 4097.345 |

本prototypeのcold取得では、OCCは同値SDPより全6Hで短いmedianを記録した。NORMは全6HでOCCより短く、目標.01の採否は同じだった。
このhost・小系・固定予算の測定であり、一般のspeedup倍率・厳密なmemory優位・量子総資源優位は認定しない。
ORは保存証拠のcoverage集計で、専用operational ORのcold取得時間を測定したとは扱わない。

## 証拠・失敗・resource監査

[独立保存監査](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/independent_saved_audit.json)は1224 certificate、36 strict反証recordを確認した。
反証は[6種類のcase/external/δ query](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/strict_refutation_queries.json)に対する理想OCC条件の非可行で、物理的縮約不能やSOS不成立ではない。
候補証拠のPSD checks4872。有理PSD/scalar/SOS残差とselector/縮約density/mode mappingを凍結独立checkerで確認し、新solver/producerを呼んでいない。

504 mode-method rowsは全て`CERTIFICATES_FOUND`。未判定・technical/soundness STOP rowは0、unverified proposals0。
SCSは432calls、`optimal`414・`optimal_inaccurate`18。18callsは固定20000iteration上限に到達した。
数値statusだけで保証を認めず、保存した有理証拠の厳密検査を経た。solver設定の変更や追加call・再試行はしていない。
method resourcesのmanaged NumPy eigh5676、別保存の共通kernel bound取得12を加え5688。SCS内部eigensolver数は未instrumented。
max monitored rational bits1417/4096。Fractionの全temporary bit長を完全監視したとは言わない。

[run summary](../../artifacts/n3_width_comparison_v2/2026-10-12/run1/result.json)のwallは71.111秒、sampled aggregate RSS peak345,907,200 bytes（約329.9MiB）、最大sample間隔.127秒、budget violationなし。
瞬間全peakや孤立method memory peakの測定ではない。一次run出力29,595,090 bytes、最大file6,485,903 bytes、出力上限内。
[出力SHA台帳](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/run_output_inventory.json)、[実行後identity/scope](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/postrun_identity_and_scope.json)、[消費済み認可STOP](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/AUTHORIZATION_CONSUMED_MANDATORY_STOP.json)を保存。

集計は結果後に用意した[標準libraryだけの抽出コード](../../artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/extract_saved_metrics.py)による、事前登録指標の保存データ抽出である。
新κ探索、新certificate、追加入力、真E0/gap、compiler、shot、N1/N2の計算は0。
旧manifest106 entriesと旧source/results/STOPは保持した。rootの`docs/progress/README.md`に外部編集を観測し、その編集を保持して本commitへ巻き込んでいない。
既存global manifestは`relevant_commits[7].description`欠落でFAILのまま。個別の証拠監査PASS、GitHub公開取得性、CI/第三者科学再現を区別する。

## 次の担当

[GPTへの引継ぎ論点](n3_width_comparison_v2_results_review_handoff_20261013.md)を準備する。新結果の研究上の価値・中心テーマ・次の範囲はGPTの判断事項。
認可は一回消費済み。追加科学実行もGPT詳細レビュー開始も自動で認可せず、Mandatory STOP、N1停止、N2保留、原primary・旧2-base gate・中心テーマnullを維持する。

固定GitHub URLとremote取得・保存照合の証跡は[結果公開記録](n3_width_comparison_v2_results_publication_20261013.md)に揃えた。
