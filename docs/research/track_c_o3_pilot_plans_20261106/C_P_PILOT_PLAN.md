# C-P pilot案：processingの有限時間信号

2026-11-06 JST。**DRAFT / 未実行・未採択。** [共通契約案](SHARED_PILOT_CONTRACT_DRAFT.md)に従う。研究Bのsynthetic2mode/N1実験を再開・再実行しない。

## 1. 問い・入力と、変更する一要素

同じPF kernelに対する物理processorの有無だけを変え、元H厳密groundからの有限時間complex signal・半径・多位相混入・必要Nの変化が、processor端点費を上回るかを調べる。

入力案は共通H4/元H ground/既存DF表現/LD3、同じδ0=.02と18round。合成profile・precision policy、tail law/tuple、PF kernelはarm間で固定する。元H/stateの対応が未確定なら主要科学評価を始めない。

純粋共役 `U_proc=P† U_kernel P` はunitary kernelの固有値集合を保存する。有限mean Qも同じ実現Pによるsimilarityなら固有値集合を保存する。固有位相bias低減を検証目的にせず、`⟨E₀|P† U_kernel^q P|E₀⟩`と`⟨E₀|U_kernel^q|E₀⟩`の違いを対象にする。

## 2. 既存資産から選ぶ具体的なkernel/P案

### 初期案：D-contextの公開YP8m8を使う一組

`b2d_adapter`にある公開YP8m8のkernel列とprocessor列を使う。H4のDContextはone-body/correction＋LD3のDF block。D内部kernelの有限反復、外側の対称D/R列、original finite-RTEを全armで共通にする。D内部反復n_Dは旧実装に存在する1を最小案として示すが、H4へ採用した科学結果はないためユーザーの選択事項とする。n_D/grid探索は行わない。

重要な区別：既存`DRun`は数学的に`P_pub K_D^n P_pub†`を**各D run内**へ配置する。一方、今回のprimaryは、固定した**全PR kernel**へ外側から同じdeterministic D-context processorを共役として一度適用する二arm比較である。DRunをそのまま連結したものが全体の`P† U_kernel P`だとは扱わない。公開係数を使っても、全PRに対する公開processed法の次数保証を得たとは主張しない。

| arm案 | 実際の定義 | 共通にするもの |
|---|---|---|
| P0 | 全interrogationのunprocessed kernel。D内部はraw YP8m8 kernel、Rは同じoriginal finite mean | D/R順序、n_D、step、r/K、合成、state、scalar |
| P1 | P0と同じkernel Qに、同じ実現deterministic Pの`P† Q P`を作用 | P以外を変えない。Pの中へ新random tailを入れない |

これは既存processorをH4 full信号へ接続する条件付き適用案。新しいPの係数探索・最適化はしない。もしD-context Pでなくfull-H公開processorを対象にする判断なら、kernel/block定義から変わるため、実行前に別の一組として明示的に置換する。

### processorの向き・列順を確定する

既存SOURCEの`published_math_sequences()`は数学的積順、native descriptorはappend順。`DRun.append_operations()`は`actual_adjoint(p_append) + k_append*n + p_append`、すなわち数学的`P_pub K^n P_pub†`を実現する。ユーザー指定の`P† U P`規約へ合わせる場合は`P=P_pub†`と定義する案。逆向きを後から試して良い方を選ばない。正しい向きとτ=h/(2 n_D)を実行前に固定する。

## 3. 物理processorと端点・RTE障壁

P1も各shotの最初は同じ`|E₀(H_phys)⟩`を受け取る。実際のP gateを作用させ、そのgate・合成誤差を費用/budgetへ含める。`P|E₀⟩`を無料の新state oracleとして供給しない。

| 境界 | 相殺できる条件 | 相殺できないもの |
|---|---|---|
| 全kernelの反復 | 同じdeterministic Pとactual inverseで`(P† U P)^q=P† U^q P`。有限meanでも同じ全体共役であることが必要 | 別P、別step、再合成した非同一inverse、randomized Pの独立drawを同一としない |
| D runの隣接境界 | 同じcontext/τ/実現列で、Rを挟まない本当の隣接inverseだけ | `P_D† R P_D`をRへ無条件交換して消さない。RTEを挟むDごとのprocessorは端点が再出現する |
| full measured wrapper | ordinary controlled `diag(I,P†UP)`を直接実装する場合は実際の両端点を数える。共通のuncontrolled system P/P†を使う測定等価loweringも候補 | 余分な操作を無料扱いしない。control0/1のoperator等価とancilla測定等価は別 |

測定等価loweringを使う場合、ancillaの最終X/Yだけを読むための共通system後処理は省略できる場合がある。その場合も**初めの物理Pは必ず実施・課金**し、省略した後処理は測定に不要なgateとして記録する。full controlled operatorとの同一性を主張せず、両枝の干渉と両X/Y期待値で等価性を検査する。primaryの端点実装はこの最適化の採否も含めて実行前に固定し、baselineにも同じ測定scope最適化を認める。

## 4. 固定SOURCE・最小差分

| 固定SOURCE | 読み替えず再利用するもの | 最小差分案 |
|---|---|---|
| [b2d_adapter/adapter.py][adapter] `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e` | `DContext`、`published_math_sequences`、`actual_adjoint`、`DRun`の向きとτ、`fuse_adjacent_d_runs`の制限 | H4 DContextとraw kernel/P endpointの接続を小さく追加。`make_b2d_plan`のper-D-run processingをprimary全体共役と混同しない |
| 同commit：`b2d_adapter/_published.py` | 元PF repository `20d64c2f…`の公開YP8m8 kernel/processor列の抜粋 | 元の係数・数学積順・actual inverseを保持。P solverを作らない |
| 同commit：`b2d_evaluator/b0_native.py` | 既存native-v5のDF/control/phase保持方法 | 固定synthetic provider/launcher/inventoryをH4用に再実行しない。必要なprimitive実装だけ適合確認して利用 |
| `f29eafbf4412193e8e8660d9f051889c2cba684b`：`rte.py`、DF builder、`finite_rte_signal_validation.py` | 同じQのcorrected/raw mean、元H stateのsignal、DF basis/control | 同じPによる全体共役のsignal入口と、一般半径/weighted会計への小さい接続 |
| [processed_cost.py][cost] `20d64c2f…` | kernel反復とprocessor_pair_countを分けた構造価格 | 構造helperはcompiled費用ではない。実際の残る端点をfull-wrapper価格へ接続 |

既存test候補は`tests/management/test_b2d_adapter.py`、`test_b2d_b0_native.py`、`test_df_rte_qiskit_builder.py`、`test_rpe_resource_accounting.py`。旧toy runner/testを再実行せず、H4/context/append順と端点接続に必要な人工検査を計画する。

既存scriptは同b190版`scripts/run_b2d_outer_b0.py`を参照する。固定synthetic scope/実行契約に入るのでH4へ流用起動しない。採用したH4 context/kernel/P一組だけを既存descriptor/native関数へ渡す薄い実行入口が必要で、旧SOURCE/launcher/STOPは変更しない。

## 5. 最小correctness・信号の検証

初期の数理/実装確認では次を分ける。

1. 同じ理想Pのunitarity、P†P、actual append順、scalar/相対phase、Pの実現inverse。scalarに数値fitして一致させない。
2. 固定ideal kernelの固有値集合の共役前後一致。差が出ればprocessor利得ではなく規約/数値/実装の問題として原因調査する。単に並べ替えた固有値labelを比較しない。
3. 元H groundから見たP0/P1の全18roundのcomplex signal、半径、共通E₀への有限時間phase、spectral weight/多位相混入。改善の有無はここで初めて観測する。
4. 同じQ有限meanに対するP共役と、actual compiled endpointsのmean/信号。pure unitary kernelと非unitary finite meanを別欄にし、Pを変えた結果でRTE誤差を隠さない。
5. ordinary controlled referenceと、採用するfull measured wrapperのX/Y。測定等価loweringなら、operator等価を要する対照と検査scopeを分ける。

まずq1と終端q131072を含むsignal評価で、単位半径仮定や短qだけの見かけの利益を排除する。小さいsector meanをcacheして全18roundへ接続する案なので、仮想量子shot samplingを初期必須にしない。Pなしのsignalが既に十分な半径を持つ場合も、不都合な点として消さない。

## 6. shot・価格・全G

P0/P1で**同じ元H/state・target/成功条件・budget policy**を使い、実半径とphase余裕から必要Nを計画する。native Pは実行した費用・合成誤差を含める。半径が良くなっても有限時間位相が悪化してbranch条件を満たせなければ全G優越の根拠にしない。

価格取得は同じtail trajectory/同じkernelをpairedに使い、P1で残る端点だけを含むfull wrapperを比較する。q1/2の価格較正とq8 holdoutが初期案。P追加費がqに対し定数であるという構造主張と、compilerを通した実価格のaffine性を別に確認する。

長qではPの端点誤差が反復q回とは限らない一方、D run内に残るPやkernel誤差を端点1組だけに数えない。先にliteral列で相殺可能範囲を確定し、価格・数値誤差modelを一致させる。大きいqの直接価格が閉じない場合は局所gain/条件付きGに限定する。

同じkernel内でP有無を比べるのがprimary ablation。実用的総費用claimには、C-Nで用いるS2/Yoshida4等の**同精度に適合するunprocessed強対照**も必要。ただしC-Nのwinnerを事後に移植せず、共通の登録対照として独立に評価する。kernelが違うことだけをPの効果にしない。

## 7. 科学call・保存物・判断

| call群 | 有限範囲案 | 保存するもの |
|---|---|---|
| 入力/kernel/P | 共通H/stateを共有。選んだD-context/τ/n_D/kernel/Pの一組、P0/P1だけ | 元H/state、published係数版、数学順/append順、Pの向き、実現inverse |
| signal | 同じkernelのideal/finite/native、P有無、全18round | complex mean、半径/位相、spec/overlap、各誤差、N/branch |
| 価格 | q1/2/8両軸、同じtail価格標本をpaired | endpointの残存/省略理由、回路/gate/depth、native誤差、n/covariance/長qscope |
| 会計 | 同targetのN0/N1とG、登録unprocessed対照への限定比較 | N削減と端点価格の別内訳、資源別G、不確かさ/未資格 |

unprocessed強対照はC-Nと同じ元H/profile/target/実現条件なら、その別途許可された固定科学結果を共有できる。条件が違う場合は、登録したS2/Yoshida4/公開高次と既存2hに限る有限対照の追加signal/価格callを実行範囲へ含める。P0/P1だけの比較で「既知PF全体より実用的」とは結論しない。

新P係数探索、両向きの事後最良選択、n_D/gridの拡張、研究B旧toyの追試、FRの再実行は含めない。

**継続材料**：元H groundの有限時間signal/十分Nが改善し、端点・合成誤差と価格を払っても全Gで有益な領域が残る。既知processorの適用知見として報告し、新固有値法とはしない。
**縮小/STOP材料**：半径/位相/必要Nに実質差がない、端点費が上回る、signalを良くするため別state oracleが必要、RTE障壁を越えた相殺が仮定される、或いは差が数値/長q価格の未知幅に依存する。負結果は選んだP/kernel/条件の範囲に限定する。

## 8. 独立性・未決事項

C-PはC-Nの新PF選択やC-Sの残差合成を待たず、同じ固定kernel/P一組・固定合成で実施可能。kernelがC-SのS2と異なるので、3pilotの絶対費用をそのまま候補順位にせず、各primary対照内の差を比較する。

**具体化済み**：既存公開YP8m8のD-context資産、P0/P1、全体共役とper-D-runの区別、向き・actual inverse、signal/shot/端点価格と縮小条件。
**未決**：共通元H/target/profile、D内部kernel/n_D/τ案の採否、Pの向きの採択、普通のcontrolled実装か測定等価loweringか、数値/価格precisionとmateriality。ユーザーが選択し別途許可するまで実装・科学callは開始しない。今回のmandatory STOPは計画公開照合まで。

[adapter]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b190fe28b98d2f94e1509e19cd3d36a63b2bb99e/src/trottertracks/b2d_adapter/adapter.py
[cost]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
