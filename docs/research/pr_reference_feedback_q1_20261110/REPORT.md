# 新A Q1：Adaptive Feedback Readoutの最小実装

2026-11-10 JST／SERVER 2。**過去countからのfeedback角生成、B′／Reference双方の任意角readout、固定340-count decoderをopt-inで実装し、18件の人工テストがPASSした。** 実H4でのfeedback、340 shotsの成功保証、追加native価格、全18roundの総費用削減は未資格である。

基準はQ0完了commit [`34327b90`][Q0]。同じ研究線のbranch `codex/algo-improve-new-a-total-cost-20261109`を継続し、新しいQ1 namespaceへ追加した。旧SOURCE・結果・STOP、既存X/Y、V6-Q producer／consumer、他Trackは変更していない。Q0科学レビューの制約は今回のユーザー指示§1から引き継いだ。独立レビュー本文を新たに取得・実施したという報告ではない。

## 1. 実装したもの

実装は[pr_reference_feedback_q1.py](../../../src/trottertracks/pr_reference_feedback_q1.py)。詳しい定義・前提は[FEEDBACK_READOUT_CONTRACT.md](FEEDBACK_READOUT_CONTRACT.md)にまとめた。

| 接続 | 実装・確認事項 | 残る条件 |
|---|---|---|
| 過去測定→prefix | m0〜16のX/Y整数countから旧branch decoderを再利用。exact count argの区間で同じbranchの最近傍性を確認。中心と半幅 `.4/t16` を条件付きで返す | 保存prefix成功event、SOURCE／law／N適合、数値検査の一様な受理保証 |
| prefix→feedback | `φ=2 arg(x16+iy16)+π/2 (mod 2π)`。binary64 parameterとの差を192-bit区間で囲む | 実測countへの新しい数値評価、実provider誤差 |
| 任意角readout | 既存未測定coreを `H→core→P(−φ)→H→Measure` で包む。両arm共通 | 実H4 coreとCompiler A contextでのcorrectness・追加費用 |
| count→confidence hull | 既存Q0の `epsilon`／`hull` を直接再利用。未知半径、旧bias、新feedback誤差を分離 | 実readoutが同じ信号モデルを満たす保証 |

旧decoderの最終出力は**基準時刻の位相**であり、m16の累積位相ではない。単純にそのfloatを2倍する方法は不適合だった。最後の測定argとbranchの整数関係を用いることで、大きな時刻と丸めたenergyの再乗算を避ける接続を具体化した。E0・gap・stateの古典情報はfeedbackへ与えない。

Prefix中心は旧float点推定を保証付きと宣言したものではなく、同じ選択branchの数学的count argを区間で復元したもの。cap<π/3の最近傍branch補題と保存alias条件の下で `.4/t16` へ接続する。ゼロvector、branch tie、alias端等の拒否を、成功確率から無償で除いていない。実測m16履歴は今回提供・生成されておらず、真energyや期待signalから代用していない。

Decoderは固定n=340のみ。追加feedback誤差が不明なら `UNQUALIFIED_FEEDBACK_ERROR`、非単調な枝なら未資格、空集合なら `INCONSISTENT_COUNTS`。観測countに応じたn・モデル変更を行わない。出力にはprior、角度／算術誤差、count、confidence区間、phase／energy hullと径、条件statusがある。保存Q0モデルを用いる接口は対応するarmの保存prefix Nへのbindingを要求するが、今回その実科学モデルで数値評価していない。

## 2. Q0完全結果の読取確認

公開commit34327b90の圧縮完全結果をbase64として全取得し、全bytesを展開・照合した。

| 対象 | bytes | SHA256 |
|---|---:|---|
| `RESULTS.json.xz` | 396,748 | `6e81eb2b8357257e5a751d3a1db930f419961554d36bb58df8ccd0b2bda2323e` |
| 解凍後JSON | 4,233,562 | `d338fb14201ec23274687a7acabe704007a53521eced2ad8c8c6c79b21259ec8` |

両armのn=1〜1252の全保存行をparseし、保存最小値340、339 FAIL／340 PASS、passing list、全count検算field、feedback margin status、旧CHECKSのraw hashとの整合を確認した。**保存原表とmetadataの整合確認であり、区間計算・全count反転を再実行した数学的再監査ではない。** Q0 campaign・checkerは再実行していない。

再利用する科学結果は[旧REPORT][Q0]の条件内に限る。両arm340、Reference-B1の十分feedback誤差条件 `.000748774257 rad`、prefix失敗予算34/720、m17予算1/360を保持する。340は登録annulus・Hoeffding・hull径判定内の十分結果。強いX/Y集合対照の最小総shot数は未確定で、341〜1252を追加探索していない。Reference固有の改善や任意推定器への最適性に読み替えない。

## 3. 人工correctnessと再利用

[人工test SOURCE](../../../tests/test_pr_reference_feedback_q1.py)／[実行log](../../../artifacts/pr_reference_feedback_q1/2026-11-10/TESTS.txt)。既存Python3.12、Qiskit1.3.0、numpy1.26.4、mpmath1.3.0の環境を読取利用し、次を検証した。

- 固定人工count履歴によるbranch・energy包含、正負位相／period cut、累積位相とbase phaseの区別、算術誤差包含、異常count・tie等の拒否。
- 人工ρ・bias・明示ideal追加誤差によるQ0反転との一致、hull包含、未知誤差・非単調枝・異なるprefix／armの扱い、失敗予算のexactな和。
- 旧B′ kernelと旧Reference core／actual inverse／vacuum関数を再利用した人工2-system-mode回路。Gaussian frame、対角位相を含む全4次元誘導observableについて、`Re(e^(−iφ)V)`との行列比較。0、±π/2、π/4、正負一般角、common global／relative phaseの違い、registerとmeasurement順を確認。
- 旧consumerのcompile→reset prefix→物理実行という順序をmock compilerで確認。実compilerは呼ばない。外部による192-bit context変更は拒否。

最終18件PASS。最初の実行で「正のbase phaseならperiod cutを越えるはず」という誤ったtest assertionが1件あり、非ゼロbranchの選択を検査する形へ訂正した。この訂正で実装は変更していない。最終test数を重複実行数で増やしていない。Codex自身の技術検査であり、独立科学レビューではない。

実H4 cache／state読込、科学角度評価、科学build／compile／simulation、sampling、保存科学値の再解析、全round再会計、native価格取得、strict合成／guard／T評価はすべて0件。SOURCEは[新SOURCE.json](../../../artifacts/pr_reference_feedback_q1/2026-11-10/SOURCE.json)、終了状態は[STOP.json](../../../artifacts/pr_reference_feedback_q1/2026-11-10/STOP.json)。共有設定・他process・packageへ変更なし。

## 4. 340 shotsの移送条件と費用

Q0の残差uはprefixの統計的不確かさを含む。これをfeedback実装誤差へ再加算しない。旧PF／SOURCE／finite-bit／native diskは保存biasとρLに既に反映されており、二重加算しない。新しい誤差は角度生成・parameter・lowering・compiler等の**追加分**として分離する。一般のoperator diskはそのままradの角度誤差にできない。

今回の算術包含と人工correctnessだけでは、実prefixの成功event、数値検査の受理率、実native Pの誤差は資格化されない。従って `34/720+1/360=.05` はこれらが成立した場合の接続式であり、Q1から実装済みRPEの成功確率≥.95を認定しない。strict Clifford+Tの追加近似誤差も未評価のままである。

Qiskit1.3 SOURCEには `P(λ)=exp(iλ/2)RZ(λ)` へのsymbolic経路がある。既にcontrolを含むwhole joint回路をlowerし、その後に新controlを付けない条件ではcommon global phaseだけの差となる。これはnative表現の存在を示すが、新角度contextのlowering経路・cancellation・数値誤差・価格を証明しない。

資源jごとに必要なm17の比較は

`340 μQ,j + Aj < 626 μX,j + 626 μY,j`。

旧m0〜16を固定する場合、全round費用は `Gprefix,j + G17,j` と記号的に分解できる。Aj、μQ,jを0・保存X/Y価格と同一には置いていない。RZ、CX、size、各回路DAG depthとその反復費、peak qubits、classical feedback latency、将来T／T-depthは別指標。異なる単位を合算しない。

両armへ同じQ機会を与える。将来のReference対B′比較はそれぞれの実signal保証と価格で行い、強Full／Truncatedにも同等のreadout・統計最適化機会を検討する。今回PR全体の優越を主張しない。

## 5. 次の判断とmandatory STOP

技術的にはadaptive feedbackの最小接続ができた。**次の最小科学検証は、保存core・有限角度に限った局所correctnessとnative追加価格のQ2**である。[NEXT_VALIDATION_PLAN.md](NEXT_VALIDATION_PLAN.md)に固定候補・call・STOPを具体化した。これだけで実m17全supportや全18round削減が資格化されるわけではない。

GPT／ユーザーが、Q2の有限scopeを実行するか、prefix受理保証と実feedback誤差の資格をどの段階で要求するか判断する。Q2は未承認・未実行。科学レビュー・追加実装・価格／全round計算へ自動進行せず、公開とremote照合後 **mandatory STOP**。

[Q0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34327b904540556bce7509f2a3168cbff36daab7/docs/research/pr_reference_q0_20261109/REPORT.md
