# 案M：動的instrumentと人工correctness

2026-11-08。**Product/Rotationの両方を実装し、最終の新規人工12件はPASS。** 実H4、旧bit/raw、RNG、shot、compileは使っていない。人工数値correctnessを全support誤差認証・科学的改善へ読み替えない。

## 実装

[m_label_average.py](../../../src/trottertracks/algorithm_codesign/m_label_average.py)の`build_instrument`は、system 0:n、ancilla nに次を構築する。

1. 入力system状態を外部から供給し、ancillaへH。
2. control0でA†、control1でB。`QuantumCircuit.inverse()`を用い、Aのgate順とglobal phaseも実際に反転する。
3. systemへVを適用し、各wireをcomputational basisで破壊的に測定する。z=Σ_j bit_j 2^j。
4. Productは共通ancilla軸を測り、出力s_a r_g(z)。RotationはQiskit `if_test((system_register,z))`でz依存軸を選び、出力s_a m_g(z)。
5. ancilla bit0→+1、bit1→−1。位相0のY軸はSdg→H。一般軸cosθX＋sinθYはRZ(−θ)→H。

prepare状態は(Vを含めて)

\[
 \frac{|0\rangle VA^\dagger|\psi\rangle+|1\rangle VB|\psi\rangle}{\sqrt2}.
\]

逆状態準備は不要。実装にstatevector、Born確率、真のsignalを渡す接口はない。builder/decoderはmetadataだけを使い、実際の量子状態は外部にある。

Rotationはc=cosφ、s=sinφに対しO_X=cX+s rY、O_Y=−s rX+cY、m=√(c²+s²r²)。共通phaseχがあるとき、Xはcosχ O_X−sinχ O_Y、Yはsinχ O_X＋cosχ O_Yを測る。両observableは同じ固有値絶対値mを持つ。Productはphaseを軸に移してrの符号をdecoderへ残す。m=0なら軸を0へ固定して出力0とする。

動的回路のinstructionは実装したが、**hardware feed-forward対応は未確認**。`feed_forward=True`はモデル上の明示指定であり、backend資格ではない。未知measurement modeを拒否し、Rotationで指定がなければbuildを止める。追加logical ancillaは一つだが、system readoutとfresh入力準備を無料にしない。新回路/decoderはbinary64人工prototypeであり、旧exact Fraction科学engineの置換ではない。

## 独立参照と検証範囲

[新test](../../../tests/tracks/algorithm_codesign/test_m_label_average.py)は人工1/2 system qubitだけを使う。独立にlabel別のB→V→signed Z/ZZまたはRZ/RZZ→V†→Aとphaseを作り、通常のcontrolled Hadamardと比較する。branch準備の全行列は既存`matrix_gate`で絶対差を確認し、phase fittingはしない。

さらに、**実際に生成されたmeasure/if_else命令**から全測定分岐を列挙する。非正規化のpost-measurement vectorで確率を保持し、reset/shot samplingは行わない。測定後に量子状態を再正規化して都合のよい結果を選ぶ処理もない。参照側だけがOperator/Statevectorを使う。

| 人工確認 | 内容 |
|---|---|
| 適格性・法則 | 非一様な有理数q、群質量、signed/identity、部分群分割、正support、grid/actual massの不一致 |
| branch | 非可換A/B、actual adjoint、global/relative/scalar/event phase、1/2-qubit wire順 |
| Product | 異なるsupport、符号、identity、共通固有値と混合固有値、X/Y平均とM2 |
| Rotation | 正/負/0角、非自明角、r=0、記号的m=0、Product→π/2 Rotationの−i位相 |
| instrument | computational readout、動的軸、Sdg→H、bit符号、decoder、不正wire/unsupported feed-forward拒否 |
| 公正な参照 | label別controlled、単独label直接測定、同じ外側lawの層別、既知の平均observable grouping、適用可能なancilla-free直接測定 |
| 実行境界 | adapterのstate/Born/RNG/loader/compiler依存なし、H4幅build拒否 |

確率総和は3×10⁻¹²以内、行列照合の最大entry差は3×10⁻¹²以内、平均/M2は小数12桁で照合した。これらは人工fixtureの検査許容差であり、一様なb_engineではない。元Hadamardと**平均は同じ**、M2は元の1と一致する必要はなく、**群平均の導出式**と一致する。

## 強いbaselineとの区別

人工|ψ⟩=|0⟩、A=B=V=I、q(I)=1/4、q(−Z)=3/4という解析例を使った。z=0でr=−1/2。以下は人工参照の二次momentであり、実H4の母分散/必要Nではない。

| Instrument | Product X / Y | Rotation φ=π/4：X / Y |
|---|---|---|
| 元のlabel別controlled Hadamard | 1 / 1 | 1 / 1 |
| label別のbranch型直接測定 | 1 / 1 | 1 / 1 |
| 案M（label平均後のbranch型測定） | 1/4 / 1/4 | 5/8 / 5/8 |
| 既知の同じ平均observable測定 | 案Mと同じ | 案Mと同じ |
| A=B†で可能なancilla-free単独label直接測定 | 1 / 0 | 1/2 / 1/2 |
| 同じ条件でのancilla-free平均observable grouping | 1/4 / 0 | 1/2 / 1/8 |

平均はProduct (X,Y)=(-1/2,0)、Rotation (1/√2,1/(2√2))で全方式一致する。**この人工例では、強いancilla-free groupingは案Mより同等以上である。** 「±1 HadamardよりM2が小さい」というだけで既知直接測定法への優位とはしない。branch型測定で群平均が加える操作はlabelの実行/取得を省いて平均observableを測る点だが、これは既知groupingと同じ統計機構であり、独立の新規性は認定しない。[既知direct measurementの一次資料](https://arxiv.org/abs/1901.00015v2)

一般のA/BでA=B†が成り立つとは限らず、上のancilla-free参照を常に適用できるとも言わない。逆に、その条件を検査せずにbranch型だけを強いbaselineと呼ばない。外側層別を両方式に同じように適用した人工例でも平均を照合した。実H4の層別分散/配分は評価していない。

## 保存結果と開発時修正

最終結果は[artificial_run03](../../../artifacts/o2_m_instrument/2026-11-08/artificial_run03/TEST_RESULTS.json)、[log](../../../artifacts/o2_m_instrument/2026-11-08/artificial_run03/RUN.txt)、[SOURCE](../../../artifacts/o2_m_instrument/2026-11-08/artificial_run03/SOURCE.json)、[STOP](../../../artifacts/o2_m_instrument/2026-11-08/artificial_run03/STOP.json)。12件、0 failure/error。runner全体2.5427秒（import含む）、peak RSS 461,208 KiB。unit test本体はlogで約0.5秒。この資源値も人工実装の一回観測であり、実H4の実行時間予測ではない。

開発run01は11件中1失敗：浮動小数の参照M2=0.9999999999999996を1と完全一致で比較したtest assertion。元平均/位相/動的回路の不一致ではなく、他の参照と同じ12桁比較へ修正した。run02はProduct→Rotation検査も含む12件PASS、run03は強いancilla-free baseline参照を追加した最終版で12件PASS。前2回のlog/hash/STOPも別directoryに保全した。run01/02は開発時のcontent hash記録であり、最終SOURCEと同一版・immutable CIとは表現しない。

既存namespace importから出る`Almost_optimal_grouping.py`のSyntaxWarningは既存のdocstring escapeに関するもので、元SOURCEを変更していない。旧A2/RTE/Parent等のsuiteは再実行していない。
