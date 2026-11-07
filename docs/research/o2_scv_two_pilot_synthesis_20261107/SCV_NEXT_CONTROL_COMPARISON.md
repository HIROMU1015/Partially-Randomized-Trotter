# 未検証control：演算子構造と実装差分の比較

2026-11-07 JST。**既存候補の静的比較。新controlの実装・人工試験・実H4数値評価・正式採択なし。**
対象は保存linear H4／1.00 Å／STO-3G／legacy DF12／8qubits・全4電子sector／L_D3／m0,q1,δ=.02,r1,K2／coefficient_atol0、N1 actual Q・selected full、N2保証ground・認証済みexact Fraction engine、理想gate・X/Y。

## 現在の状態への更新

[4741caeの比較][old]ではSigned Supportは実H4未検証だった。その後、[metadata結果b4512675][meta]がh1非定数性を確認し、[448-shot結果a7b70ff][signedresult]がfresh fit/評価と学習費込み会計を完了した。今回この状態を反映し、Signedを未実施候補として再提案しない。Product一致・products sign積・basis群等は、既存設計の段階に留まる。現在の2pilotによる候補間の科学的順位は付けない。

以下の式・候補は[84e021b6の既存候補設計§2–6][candidate]の再掲。ρ(l)は保存53-bit actual label質量、a≠bはN1の固定label。適用順products→rotation、n=2のproductsはL0,L1、rotationはL2。

| h | 固定条件 | 各位置の条件付きlaw |
|---|---|---|
| 0 | n=0,L0=a | δ_a |
| 1 | n=0,L0≠a | ρ_¬a |
| 2 | n=2,L0≠a | ρ_¬a, ρ, ρ |
| 3 | n=2,L0=a,L1≠b | δ_a, ρ_¬b, ρ |
| 4 | n=2,L0=a,L1=b | δ_a, δ_b, ρ |

各位置は層内で独立。ρ_¬a(l)=ρ(l)1{l≠a}/(1−ρ(a))。これは既存[ConditionalSampler.draw/_pick][sampler]の条件付き整数取得に対応し、新たなQの採択や全path列挙を必要としない。

## Product label一致：異なる構造を問える候補

\[
u_{\rm eq}=1\{n=2,L_0=L_1\},\quad
\nu_{0,eq}=\nu_{1,eq}=\nu_{4,eq}=0,
\]
\[
\nu_{2,eq}=\frac{\sum_{l\ne a}\rho(l)^2}{1-\rho(a)},\qquad
\nu_{3,eq}=\frac{\rho(a)}{1-\rho(b)}.
\]

indicatorなのでm₂,h=ν_h、Σ_h=ν_h(1−ν_h)。式はexactな有限平方和・条件付き比で表せる。**Product一致の実H4数値表は未取得**。signedの保存表や旧3-controlの平均を代用しない。

- **CONFIRMED（既存law/SOURCEからの構造）**：h2では自由なL1がL0と一致/不一致になり得る。h3ではL0=a、L1≠bで、a≠bのためL1=aも別labelも可能。既報の正label supportから両層に変化する余地がある。h0/1はn=0、h4はa,b固定でu=0。新たに実catalogを集計した主張ではない。
- **CONFIRMED（演算子関係）**：同じ完全labelの同じsigned involutionをP_lとすればP_l²=I。固定signも二度で相殺する。異なるDF基底の同じZ文字列を同一Pと扱わない。[DF spec][dfspec]と[RTEEvent][rte]に対応する。
- **HYPOTHESIS**：productsが恒等化すると条件付きHadamard平均を区別し、bitとの相関が生じる可能性がある。
- **UNKNOWN**：実際の相関・有限学習係数・母分散低下・全RZ改善。rotation、n=2のevent phase、scalar/identity phase、決定論Dブロック、controlled diag(I,U)、Hadamard wrapperは残る。**productsの恒等化をfull interrogationの恒等化や決定論bitへ読み替えない。** [既存builder][builder]

### 検討意義と費用上の弱点

Fragment群とSignedはいずれもrotationの一位置情報をh1で学習した。Product一致はproductsの二位置の厳密な代数関係を使うので、同じcontrolの単なる追加測定とは異なる問いになる。path metadataの2label比較に追加量子測定は要らない。平均取得はlabel数に線形の平方和で済み、実H4全path列挙・Born計算を不要にできる。

一方、h2/h3は既報N1の希少order-2層。条件付き取得に量子棄却の1/p_h倍費用は不要でも、独立trainの各shotにはfull RZがかかる。全targetへの追加削減は既存式のΣ p_h²Δ_h/n_hで重み付けされる。大きい層内相関があっても全体の小さい利益しか生まない可能性がある。小さいp_hだけで一般的無効とも断定しない。**この弱点を調べる動機がなければ、新しいcampaignを実施する理由にはならない。**

## その他の既存候補との比較

下表は構造・再利用負担の比較で、実H4数値性能の順位ではない。

| 候補 | 活動層・known mean | 構造上の理由とUNKNOWN | range・学習/古典費・検討価値 |
|---|---|---|---|
| 現Fragment群 | 実装はh1活動。ν/moment既得 | 同fragmentのbasis族を区別。具体Sだけ224完了、族全体は未知 | [0,1]、D=2+abs(λ)。追加は同controlの別の問いを定めた場合のみ |
| 現Signed Support | 実装はh1活動。全5層ν/moment既得 | 符号/support族。構造的ELIGIBLEと448完了、効果保証はない | [−1,1]、D=2+2abs(λ)。448を未実施状態へ戻さない |
| Product完全label一致 | h2/h3。上記平方和/比、u²=u | P_l²=Iという別構造。bit相関・全費用への寄与は未知 | [0,1]、D=2+abs(λ_h)。活動層ごと有料train。将来候補の第一比較対象だが未採択 |
| products sign積 | h2/h3で変化し得る、h4固定。条件付き一位置sign平均の積 | signed product phaseと関係。全wrapperの信号は予測しない | [−1,1]、D=2+2abs(λ_h)。2lookup。Product一致と同じ希少層の弱点。両方同時投入を提案しない |
| 保存basis群のindicator | 自由rotationならh1/h2–4で変化し得る。群質量の一位置和 | 演算子基底族を区別するが、fragment群と情報が重複し得る。bit相関未知 | [0,1]のscalarなら低負担。具体群の事前固定・新mean・独立trainが必要 |
| 局所操作数の加法proxy | 自由label間で値が違う場合。有限一位置和、隣接項なら二位置和 | 価格予測とbit説明力は別。真のcompiled RZの平均ではない | 事前の有限box/scaleが必要。古典lookup。信号との新しい根拠なしに実装を増やさない |
| 特定label・固定label組合せ | 既存LocalControlの積law・包含排除 | 稀な出現と学習rank、応答との関係が未知 | 既存API再利用性は高いが全label one-hot化は学習費を増やす |
| Taylor次数・基本角・event phase | 元5層内で定数 | 新しい層内変動を説明しない | 新SCVには進めない |
| full compiled RZそのもの | 条件付きexact平均が安く既知とはいえない | compile・未知平均の別費用が要る | 無料known-mean controlではない |

Born確率、真のpath信号、state vector、E0/gap、真variance、測定word/bitをcontrolへ無料で与えない。候補の数を増やした多変量探索や旧bitでの再ランキングは今回の比較に含めない。

## Product候補で再利用できるSOURCEと必要差分

参照SOURCEは固定`f70e555fbfe08676f530d989192aade75a57ab1a`、indicator核は`560c6092bfb5e71af8eedc4a4be5416922764f98`。既存ファイルは変更しない。

| 再利用するもの | そのまま使えない点／将来の最小差分 |
|---|---|
| ConditionalSampler、PathChoice、Domain/Streams、validate_batch | 元5層/Qを維持。新controlは元層IDと別identity。新stream表とpurposeを明示 |
| MetadataのLAW/catalog byte・H/preparation binding | 新しい完全label一致feature/meanのtyped bankが必要。label文字列だけでbasis/sign対応を省略しない |
| ControlBank.joint_mean/pattern_mass、LocalControl | 特定のlabel対の平均は扱えるが、「任意labelが互いに一致」は単一LocalControlではない。平方和の限定adapterが必要 |
| fit_and_freeze/FrozenCVのindicator算術 | OLS core/中心化/rangeは再利用できる。`scv_scalar.CellModels/fit_cells`と`SignedFrozenCV/fit_h1/validate_training`はh1専用で、h2/h3へそのまま移せない |
| paired_evaluateの加重分散・covariance算術 | 独立trainを複数活動層へbindするmodel/antecedents/serializerを限定拡張。全5層を維持 |
| LegacyCollector・cost_summary・SavedPrices | 448固定quota/CALL_CAPS/seed/halfを継承しない。目的に合う新固定表を呼出し側で用意 |
| qualified manager・observer・journal・reconcile、signed runtimeの短いcall admission/STOP | 科学scope・SOURCEと新output/cache専用bindingが必要。旧job/STOPを再利用しない |

SOURCE参照とblob照合は[EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json)。人工correctness、metadata-only、fresh検証の最小引継ぎは[次作業草案](NEXT_MINIMAL_VALIDATION_DRAFT.md)。本書を理由に実装・テスト・科学計算を自動開始しない。

[old]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4741cae05af662dfc42d0519ff977756a94cb7b7/docs/research/scv224_next_directions_20261107/SCV_CONTROL_CANDIDATE_COMPARISON.md
[candidate]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/INTRASTRATUM_CONTROL_CANDIDATES.md
[meta]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4512675ae1d650858256547d359210361d77f68/docs/research/scv_signed_support_h4_metadata_science_20261107/REPORT.md
[signedresult]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/REPORT.md
[sampler]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f70e555fbfe08676f530d989192aade75a57ab1a/src/trottertracks/algorithm_codesign/a1_n3_connection.py
[dfspec]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f70e555fbfe08676f530d989192aade75a57ab1a/src/trotterlib/df_rte_circuit.py
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f70e555fbfe08676f530d989192aade75a57ab1a/src/trotterlib/rte.py
[builder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f70e555fbfe08676f530d989192aade75a57ab1a/src/trotterlib/df_rte_qiskit.py
