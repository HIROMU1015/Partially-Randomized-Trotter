# 最初の共同最適化検証案（未実行）

[統合調査](REPORT.md)／[条件対応](INTEGRATION_MATRIX.md)／[再利用SOURCE](SOURCE_INDEX.md)。**今回の許可は計画作成まで。本書は次の科学実行許可ではない。** 新しいN・総資源・最良候補・削減率は算出していない。

## 1. 第一案 I0：固定CDF8のα配分×control方式

目的は、既存元Hの誤差証拠、十分shot算術、実装方式ごとの価格、全round集計を一貫した候補IDで接続できるかを、小さいscalar campaignで検証すること。Referenceが常に勝つこと、PRが強PFより安いこと、global optimumは目的に含めない。

### 固定する科学条件

- Linear H4／隣接1.00Å／STO-3G／Nα=Nβ2、元Hの厳密ground仮定、状態準備費除外。
- 既存packet／number-CDF8／元frame順LD2、ordered S2/s1/r32/K0、h=δ=.1、C1 OFF。既存basis共有・step融合・scalar/identity・actual inverse・χを保持。
- 13round m0..12、q1..4096、εE=.00159 Ha、failure=.01、βtotal=π/6、βstat=π/12、energy prior[-10,10]、既存decoderとφ=−hE規約。
- B1/B4の元H corrected operator誤差・guard・attenuation・正半径・phase余裕・ε_coordinateを凍結。ground/DF/新PF誤差を求めない。ideal finite-RTE IID/exact-nativeという資格を維持。
- 8828b298の保存価格recordとA/F/S仕様をそのまま入力にする。q1 event16 cut、q≥2中央step cut、P0 Qiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free。追加q64結果で旧modelを更新しない。

### 変更変数と有限比較集合

最初の科学変数は**round別の失敗確率配分α_m**だけ。X/Yを各roundで同じα_mに束ね、`2Σα_m≤.01`を課す。これにより既存のequal-X/Y shot会計を使える。control方式は既存Strong symmetric-directionalとwhole-path Referenceの2種。law、PF、cut、r/K、β、measurement familyは変更しない。

候補は次の最大5配分を事前固定する案である。

1. 保存一様配分 `.01/26`。
2. 主scenario AのStrong価格から作るRZ感度配分、CX感度配分。
3. 同じAのReference価格から作るRZ感度配分、CX感度配分。

重複する配分は値の同一性で除き、各方式に**同じ候補集合の和集合**を与える。一方だけ有利な配分を許さない。RZ/CXは別々の提案座標であり、正式な主資源の採択・加重合算ではない。全候補の6累積資源とpeak qubitsを保存する。depth/sizeも事前に同じ表で比較し、都合のよい指標へ事後変更しない。

提案式は既存rpe_allocation_sensitivity_validationの `w=2C/ε_coordinate²` を再利用し、X/Yの等配分制約下ではpaired価格 `C_X+C_Y` をround重みに用いる。正値・finite・exactなα和を確認し、丸めでfailure予算を超えないよう既存の保守的な調整を用いる。これは連続最適性や整数Nの最適解を保証しない。

同一配分なら両方式のNは共通にできる。しかし方式ごとに別配分を選んだ後のNは異なり得る。同じenergy・総failure・誤差条件を満たすもの同士を比較し、異なるNを旧common-Nの違反や無料shot削減と解釈しない。

### SOURCE・入力と最小実装

| 用途 | 再利用する固定資産 |
|---|---|
| 元H誤差・coordinate余裕 | B1結果d3913866、B4-2結果07d18bfeのrun01/result.json（13行のepsilon_coordinate_X/Y等） |
| 保存一様shot・decoder | 同B4 Stage_C_handoff.jsonとaxis_shots.csv。総14,738はbaseline照合専用 |
| N算術 | rpe_resource_accounting._hoeffding_shots、c1_stage3_accuracy._assess_raw_signalの既存座標幾何 |
| 配分案 | rpe_allocation_sensitivity_validationのcost感度式。全validatorはproviderを呼ぶので実行しない |
| 単価 | 8828b298のA/F/S prices52.json、SOURCE/input_bindings、保存SCENARIOS。新price_records生成・fitは不要 |
| 総資源 | c1_stage3_accuracy.aggregate_totals。equal X/Y計画と新candidate IDを接続 |
| 同値性 | B6bのcommon_N条件とideal測定同値性。条件をTrueと書くだけで外部事実の証明にしない |

**既存guardを解除しない。** OriginalHTarget/assess_energyは全roundで同一alpha_axisを検査し、B4 read_b4は固定N・targetを検査する。Cのallocateも18round/.05/weighted資格へ固定されている。どれも新配分を型だけ変えて押し込む用途ではない。

次回必要なのは、新namespaceの小さい「allocation表→十分N→共通energy/failure条件→cost」のadapterである。元H shared targetとallocationを分離し、保存εを使うshot検査・α和・全round coverage・alias/decoder不変性を検査する。古いB4結果やtarget IDを改変せず、新しいSOURCE/input/policy IDで識別する。判定を通すために `same_accuracy=True` を無条件注入しない。

### 検証順序と新規科学計算

1. 静的binding、対象ID、全26軸・52 method/axis price、6資源、state/normalization/compiler/cutを確認する。人工整数/mockで配分の和、ceil、候補共通性、価格sample数を倍率にしないことを確認する。
2. 保存一様αでNと8828b298のA/F/S会計を再現する。これは**次回許可が必要な数値再集計**であり、今回実施していない。必要なscalar/丸めの既存規約を維持する。
3. 固定最大5配分について、新しい十分Nを保存εから算定する。新しいPF/RTE誤差やstate-vectorは評価しない。各配分で同じphase/energy幅と総failureを満たすかを検査する。
4. A/F/Sの凍結価格に同じ全候補表を適用し、資源別G、旧一様からの差、control方式差、shot数とpeak幅を保存する。Aで作った候補をF/S結果を見て作り直さない。
5. policy別・モデル別に結果を保全し、SOURCE/input前後照合、公開ref/tree/bytes確認後mandatory STOPする。

必要な新科学計算は、保存scalarを使った有限のN再算定と13-round資源会計だけ。sampling、回路生成/compile、DF fit、ground solve、PF誤差評価、新signal simulation、A/F/S fit/bootstrap、量子backend shotは不要である。scalar表が小さいため、無意味な大量workerは作らず、実空き資源に応じた通常の実行でよい。

### 判断指標・成功／STOP条件

I0の技術成立は、baseline再現、全candidateの元H/phase/alias/failure条件、同じ法則・同じ比較機会、6資源と幅の一貫した会計で判定する。改善率の数値閾値や新たな主資源は今回採択しない。モデル内で一様policyより低い値があっても、conditional estimateとして扱い、母平均優位・actual成功・PR対強PF勝利を認定しない。

入力hash／target／state／SOURCE不一致、baseline不一致、非正半径・phase余裕不足、不正α・coverage不足、有限でないN、guardを破らないと接続不能、意味の異なる単価混在は依存処理STOP。技術的修正を許可する場合も初回失敗とSOURCEを保全し、科学条件・閾値・候補familyを結果に応じて変えない。

A/F/Sで順位や方向が変わればモデル感度として報告し、都合のよいscenarioを正式結果にしない。CI不足やmodel biasを0で補わない。改善が小さい／ない場合も一回の有限検査として終了し、h/r/K/LDのgridを自動拡張しない。

## 2. 次に接続価値が高い研究案 I1（独立・未許可）

I0は最小の配線検査であり、新機構の研究ではない。研究上の追加効果を問う次候補として、**新AのQ readoutをClosed P3と強Full／Truncatedにも公平に接続する有限資格検証**を挙げる。Q3 canonical/B′/Reference-B1の検証は既に終了しているため、その再実行は提案しない。

まず既存DF12／C-O3の保存policyを固定し、支配round一つで、各arm自身のprefix区間・bias・radius/range・finite-bit/native diskがQの一様branch/角度/受理条件に入るかを調べる。Closedのweighted meanと±1 physical readout、全path weight、一般native誤差のvirtual-state移送を分離する。E0や他armの測定履歴をfeedbackへ渡さない。Q3の340 shotsを転用せず、同じtarget/failure下で必要な十分shotを各armに資格付ける。

Referenceを最新Closed/joint basisへ加える場合は、whole Vのvacuum ray・χ・actual adjoint・cutで失う融合・compiler contextを別に確認する。n=4 sectorや一つの保存pathの一致だけでは全path条件を保証しない。Cのserial価格、Q3のnative境界、D8のP0 A/F/Sを混ぜない。

この一roundの条件が閉じる前に巨大m17 full-wrapperを生成しない。一般補題／既存context索引／小人工fixtureを先に使い、必要な数値bound・有限価格取得・compileの上限は次の個別契約で固定する。成功しなければ共同採用を保留し、独立の既存Q/C成果を保持する。自動的な全18round Q再最適化はしない。

## 3. さらに別案 I2と研究判断

元Track AのA3（H4 LD4–7）やK3（D/R/drop）は既存調査にある未実行案で、別の意味を持つ。特にLD変更ではλ_R、D/PF、range、native価格、Reference成立条件が変わるため、D8のA/F/Sや保存Nを使えない。h/resolution/αの既存C menu再探索とは区別し、元H boundと強Full/Truncatedを維持した別scopeで検討する。

**ユーザー／GPTが次に決めること：** I0の限定scalar接続を行うか、研究目的を明確にしたI1/I2を先に行うか。実行する場合はSOURCE、固定入力、変更変数、有限比較集合、主張水準、技術修正/one-shot条件とSTOPを明示して許可する。正式な主資源・資源間重み・科学的GOは今回採択しない。

**Codex：現在はmandatory STOP。** 本書の計画を自動実行しない。既存SOURCE・grant・raw・ledger・marker・KEEP・STOPと共有環境を維持する。
