# 新A V6-Q2：保存4構成のexact binary phase correctness

2026-11-09、SERVER 2。**FOUR_SAVED_FIXTURES_PASS_ALL_SUPPORT_PENDING。** 固定4構成のQ2科学campaignを1回実行し、全構成でexact和・合計RN64・SOURCE相対位相・X/Y・vacuum・actual inverseの限定検証がPASSした。Q3、m17 whole、compile、sampling、新価格、RPE再計算、総Tへは進んでいない。

この結果は**有限4構成のSOURCE correctness**であり、m17全finite-bit supportの資格やV6-Pの条件付き総RZ利益を実装済み結果へ昇格するものではない。

## 1. 固定条件・実行SOURCE

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0／compiler A／rz,sx,x,cxを保持。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・二つのCSWAP cascade、D融合・basis共有・scalar／signed event／X/Yも不変更。energy精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis 1/720、保存RPE scheduleを変更していない。

| 固定資料・SOURCE | 内容 |
|---|---|
| [Q2計画 eaf70d88](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/docs/research/pr_reference_v6q_preparation_20261109/Q2_VALIDATION_PLAN.md) | q1/r1、q4/r1×rare/nonrare、各j0。negativeはq1 nonrareのancilla Zだけ |
| [provider eaf70d88](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/src/trottertracks/pr_reference_v6q.py) | Q1実装をbyte不変更で使用。旧provider default／m17 opt-inの選択規則も変更しない |
| [実行driver ca19707b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ca19707b5c0c1cefa338d47987c27f338cb4cf67/scripts/run_pr_reference_v6q_q2.py) | 科学実行前にcommit／公開・byte照合。人工driver 5テストPASS |
| [元SOURCE f2638a78](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f2638a7845685a119619382b850441f5d8d33713)／[4入力manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/artifacts/pr_reference_v6q/2026-11-09/preparation/Q2_INPUTS.json) | 既存event、common cache、pilot／transfer／B′／compiler・builderを再利用。新solve／DF／samplingなし |

共通cacheを既存hashと照合して自分のfresh outputへ配置し、`common()`のread-only loaderを使用した。旧initialize／attach_cache／科学runnerは呼んでいない。保存groundは検証stateにだけ使い、providerへE0・gap・state情報を渡していない。

## 2. 実行範囲と来歴

固定順はq1 nonrare→q1 rare→q4 nonrare→q4 rare。各構成で既存`reconstructed`と`assemble`から同じeventsのF/Gを構成し、既存`build_saved`による強B′をcorrectness参照とした。新providerは各構成1回、計4回。旧／新ReferenceのX/Yは計16 axes、negativeの2 axesを加え18 axes。各laneのvacuumは計8、FのSOURCE inverseは4構成で確認した。17-qubit whole Operatorは作らず、小さい8-qubit laneのOperatorを使用した。

2つの独立processで旧／新X/Yを並列評価し、fixture間は失敗時STOPを守る固定順で処理。小さい2×2／4×4 statevector行列積の過剰なBLAS team生成を避けるため、このprocess群だけOPENBLAS_NUM_THREADS=1を使用した。CPU／RAMの恒常上限や共有設定は変更していない。起動前MemAvailable 488246552 kB、CPU／memory PSI avg10=0.00、affinity64 CPUを観測した。

実行前のGit commit同期処理に不備が1件あり、CLIのSOURCE lookupで停止した。この時点では出力領域も作られず、科学入力読込・provider・回路・simulationは0件。Git metadataだけを同期し、同じdriver／provider／固定条件で開始した。**CLI起動は合計2回、科学campaignは1回、科学retryは0回**として記録する。科学campaignはexit 0で全4構成を完了した。

## 3. 有限correctness結果

閾値は旧H4／transfer規約の**絶対誤差1e-9未満**を保持。phase fit、joint／branch別の事後phase補正、stateやsignalの事後renormalizationを行っていない。

| q/r・stratum | 保存seed | 旧Reference X/Y複素誤差 | B1 Reference X/Y複素誤差 |
|---|---:|---:|---:|
| 1/1 nonrare j0 | 4216572573356262833 | 8.843712154383355e-15 | 5.551115123125783e-17 |
| 1/1 rare j0 | 4841711708811669773 | 2.0066754507859474e-15 | 2.931015397124743e-16 |
| 4/1 nonrare j0 | 7349747399721085684 | 2.2998119888269128e-14 | 3.3422138886441676e-16 |
| 4/1 rare j0 | 12931531744631036724 | 7.492321902858974e-15 | 1.0345236923802117e-15 |

参照zは同じ保存eventの強B′ SOURCE coherence。旧Referenceも全4構成で既存閾値を通った。新旧の小さい誤差差を一般的な科学性能差やRPE成功率の測定と解釈しない。

| 確認項目 | 結果 |
|---|---|
| exact binary入力 | builtins.floatとnumpy.float64を受入れ。g／RZ/RZZの値・順序・wireとglobal_phaseは読取前後で不変 |
| F/G合計一回丸め | 独立の整数common-grid参照と一致。返却値を有理数中点・偶数tieで再検算し4件PASS。途中のF/G別float丸めなし |
| P転送・core構造 | 17-qubit旧／新coreはancilla relative Pの角以外が一致。P parameterのhexは返却floatを保持、CSWAP cascade・gate順・registerは不変更 |
| vacuum leakage | 8 laneすべて0.0（有限数値診断） |
| vacuum action誤差最大 | 1.0026641665932502e-15 |
| F SOURCE inverse誤差最大 | 3.8547886161109927e-14。compiled CT wordのactual adjoint資格とは別 |
| fixed negative witness | q1 nonrareのancilla Zで−zへ反転。−zとの差0.0、正しいzとの差1.9998740293065511。必要なrelative位相をcommon scalarとして除けないことを検出 |

SOURCE構成・型／param・P転送について未確認だった部分を、**この4 fixtureの範囲で**実測確認した。overflow／subnormal等のrejectはこの4件では発生しなかった。これを全supportでrejectが発生しない証明とは扱わない。

## 4. 保存結果・post-run検算

- [SOURCE.json](../../../artifacts/pr_reference_v6q/2026-11-09/Q2_run01/SOURCE.json)：実行／provider commit、hash、4入力manifest、環境観測。
- [RESULTS.json](../../../artifacts/pr_reference_v6q/2026-11-09/Q2_run01/RESULTS.json)：4件のphase・exact和・RN証拠・coherence・vacuum／inverse／negative結果。
- [LITERALS.json.xz](../../../artifacts/pr_reference_v6q/2026-11-09/Q2_run01/LITERALS.json.xz)：実F/Gの全gate・wire・binary literal・global_phase。SHA256 `e79cd996d3e550e5d07007af825840bd1e3279112022596ec8e03cfb7d48b606`。
- [CHECKS.json](../../../artifacts/pr_reference_v6q/2026-11-09/Q2_run01/CHECKS.json)／[検算SOURCE](../../../scripts/check_pr_reference_v6q_q2.py)：保存literalからexact和を独立再構成し、RN中点、閾値、4件coverage、raw hash・cache・frozen SOURCEを確認。provider・builder・simulationを再呼出ししていない。
- [STOP.json](../../../artifacts/pr_reference_v6q/2026-11-09/Q2_run01/STOP.json)：4件完了、Q3未開始、mandatory STOP。

post-run検算は全PASS。RESULTS SHA256は`7cbb4c6b0af3dbeb8e3859715aa1a321d1f46906abc338cf8cae8027cedaa141`。各fixtureの個別JSONも保全した。共通pickleは同じbyte hashでlocal cacheへ保持し、公開時は元の固定archiveを参照する。旧SOURCE・raw・結果・grant・ledger・marker・STOPと既存準備資料を変更していない。

これは著者自身の実装／数値／保存結果検算であり、別担当による独立科学レビューではない。

## 5. V6-Pへ接続できる範囲と残る障害

Q2では、実Qiskit SOURCE literal、numpy.float64を含む入力型、exact F/G合計、一回RN、actual SOURCE inverse、scalar／signed eventを含む相対位相、P角のbit保持、X/Y整合を固定4例で確認できた。旧compilerのsymbolic P→1 RZ資格に必要な**SOURCEではP角だけを変える**という前提を、この範囲で支持する。

compile・新価格取得は0件であり、全m17 native価格を新たに取得・検査した結果ではない。既存固定compilerのcontext／pass／parameter形状を保つ構造的論証は引き続き利用するが、有限4例を全supportの証明へ置き換えない。

残る主な証明義務は、m17 canonical finite-bit全supportでの入力型・finite literal域とprovider受入れ、非零subnormal exact和のrejectが起きないこと、保存M／B_normの同じSOURCE経路への適用、m17だけのopt-inを実saved schedule callerへ結ぶこと。今回の4例のdyadic分母や位相値だけで全supportのlatticeを推定しない。classicalな長round走査費も未評価。

従ってV6-Pの626 shots・総RZ削減は**引き続き条件付きの既存結果**であり、今回再計算・昇格していない。旧normalizationや他diskを消去せず、m17 CT、domain／emitter／word、D3、新合成、総T／T-depthのSTOPも維持する。

## 6. 完了・次担当

承認された固定4構成のQ2を完了し、結果・SOURCE・検算・STOPを専用results branchへ公開する。remote SHAと全成果物の内容を照合後、**mandatory STOP**。追加fixture・新compile・RPE／価格再会計・Q3科学検証へ自動進行しない。

次担当はGPT／ユーザー：この有限PASSを踏まえ、m17全support受入れ・normal域・同じM／B_norm適用を閉じるためのQ3範囲を判断する。保存literalの新規数値監査等が必要なら、対象・SOURCE・判定規則を具体化して明示承認を受ける。今回その監査や追加研究は実行していない。
