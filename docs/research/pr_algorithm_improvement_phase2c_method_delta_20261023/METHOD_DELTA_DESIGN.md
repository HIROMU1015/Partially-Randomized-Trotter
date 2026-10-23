# 限定追加案：反可換Pauli対を一つのinvolutionへまとめる

2026-10-23 JST。**監査結果は独立方法差なし。** 一つの追加案について、量子実行単位そのものを変更する手順まで具体化した。その中核は既知unitary partitioningの二項の場合であり、PRへの接続は既存の一般involution-RTEを用いる通常の組合せだった。新算法として採択しない。新規実装・科学計算・非自明な新証明は実施していない。

## 1. 変更対象と固定するtask

Phase 2Cの一つの固定出力辞書を入力とし、そのselectorは再設計しない。残差を

\[
R=R_S+\sum_{i\in\mathcal P} a_i P_i
\]

と書く。`R_S`は変更しないframe側、`P_i`は物理registerのunsigned Hermitian Pauli word、`a_i`はcollect済みの非零実係数。Iは既存policyの置き場所を維持して対象外とする。D/R帰属、物理H・sector・state・scalar、DF input/rank、q/r/K、精度と測定taskは固定する。

変更するのはP側の二atom `aP+bQ` を一atom `ρA`へ置換する生成規則。単に候補を異なる順に評価する案ではない。rawのtrajectoryを見てから演算子を並べ替える案でもない。一般にはAが元のP atomと異なるが、入力によって既存frame atomと一致する場合や、P側に適格な対がない場合もある。全入力で新候補が増えるとはしない。

## 2. 入力、取得情報、出力

必要情報はPhase 2Cが既に持つPauli word、実係数、register／scalar identity、既存RTE設定、Pauli回転recipe。反可換性はPauli labelのbinary symplectic積で判定できる。DF frame間の反可換性を浮動小数点のdense行列で探索せず、S側は変更しない。真のE0・gap・ground state・新しい価格oracleは不要。

出力recordは `PauliPairAtom`：元ID対、係数a/b、ρ、ordered Pauli productの±iを含むgenerator、角度φ、実装recipe版、scalar／precision／sector metadata。unmatched atomは元のまま残す。新atomは `coefficient_abs=ρ, coefficient_sign=+1` とし、a/bの負符号はAの中へ一度だけ入れる。DF fragment IDを偽装しない。

## 3. 変更規則

Pauli対が厳密に反可換、`{P,Q}=0`である場合に、既知のunitary partitioningの式を用いる。

\[
\rho=\sqrt{a^2+b^2},\qquad
A=(aP+bQ)/\rho,\qquad aP+bQ=\rho A,\qquad A^\dagger=A,\quad A^2=I.
\]

これらは[Izmaylov et al., v2 §II.A, Eqs. (7)–(9)](https://arxiv.org/html/1907.09040v2)の二項の特殊例であり、本研究の新定理ではない。元の2componentを残したままAを追加すると二重計上になるので、置換はdisjointな対に限定する。

対の選択はstable-ID順で、未使用のiに対して最初に見つかる反可換な未使用jを組にする。最大matchingや費用最適matchingは求めない。これは構成を一意にする通常の規則であり、新しい方法差の根拠にはしない。

```text
pair_dictionary(frozen_tail):
    validate full-register labels, real coefficients, scalar/zero policy
    keep S atoms and identity at their existing locations
    scan collected nonidentity Pauli atoms by stable ID
    for each unused atom i:
        j = first unused later atom with exact symplectic anticommutation
        if no j: emit original i
        else:
            mark i and j used
            emit PauliPairAtom(i, j, rho=sqrt(a_i^2+a_j^2),
                               generator=imaginary_unit*P_i*P_j,
                               phi=atan2(a_j,a_i), recipe=known_unitary_partitioning)
    rebuild canonical RTE components, lambda, order law, angles, B
    emit ordered circuit dispatch plan; never reorder sampled products
```

係数・φ・ρを今回数値生成していない。zero／符号／精度が未確定ならその対を不適格とし、元atomへ戻す。反可換対がなければ元辞書を返す。途中で高価な汎用unitary最適化や別group探索へ拡張しない。

L個のP atomをN qubitのbit列で持つなら、素朴な候補走査は二乗個のpair判定とword長に応じたbit操作を要する。graph全保存を必須にせず、stream走査とused集合で構成できる。係数map取得費は元Phase 2Cから継承し、matching、precision、sampler再構築費も別に数える。poly規模のlabel処理だけで総取得費が安いとは言わない。

## 4. actual circuitへの写像

既知のsequence-of-rotations構成を二項へ制限する。generatorを `Y_PQ=iPQ`、`φ=atan2(b,a)` とし、

\[
W=\exp[-i\phi Y_{PQ}/2],\qquad W A W^\dagger=P,
\qquad A=W^\dagger P W.
\]

orientationと負係数のquadrantを含むこの規約は[Ralli et al. (2021), §II.A, Eqs. (11)–(16)](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033195)の二項への対応である。新証明を行った意味ではなく、実装のsign・adjoint認証は未実施。

child rotationの時間順recipeは `W → controlled exp(-iθP) → W†`、product Aは `W → controlled P → W†`。Wはsystemのみの非制御Pauli回転、中央とevent／scalar位相はancillaへ正しく接続する。Wを測定・postselectionで確率実装する案は採らない。全gate列のactual adjointを使い、a/bの符号を外側で再適用しない。

既存toy `Primitive` の `basis` はGaussianに型制限されず、任意の `QuantumCircuit` を受ける。従って `basis=W†, label=P, coefficient=ρ` がこのAを表す。`Primitive.apply`の時間順basis.inverse／central／basisにも対応する（`C-CONSTRUCT`, 約143–166行）。typed production adapterとprecision検証は必要だが、toyに新しい抽象involution型が必要なわけではない。DF production builderへWをGaussian basisと偽って渡さない。

## 5. PRへの接続と改善機構の実体

新辞書でもRを保持する理想条件下では、λとp、`τ=λδ/r`、paired次数law、θ、Bを更新する。各eventのproductと最後のchildの順序・符号・relative phaseを維持し、元と同じ有限Taylor polynomialへ接続したい。必要条件は[意味論](SEMANTICS_AND_OBLIGATIONS.md)のPOへ残す。

atom数は減り、対の係数質量は既知の関係 `sqrt(a²+b²) ≤ |a|+|b|` になる。しかし、q/rが固定なら**childの個数は減らない**。Pを選ぶchildがAを選ぶchildへ変わり、中央操作の両側に係数依存W/W†を追加する。従ってchild単価が構造的に安くなったとは言えない。RTE normalizationの負担を回路生成費へ交換する、既知の表現tradeoffである。

同じAが連続すればW†Wを相殺できるが、これは既存同basis共有に相当し、両対照へ与える。非可換列をpair別にsortして長いrunを作らない。高次数parentが稀でもAのchildは実行されるため、追加Wは典型eventにも乗る。

## 6. 方法差が成立しない理由

1. 正規化した反可換和をinvolutionにする構成はunitary partitioningそのもの。
2. Wによる対角化とコヒーレント回転recipeも既存一次資料に具体式がある。
3. 既存PRの一般involution条件と `C-SAMPLER` を使うため、PR固有のfinite event／child-productの新構造を付け加えていない。
4. 同じ対集合を与えた既知構成は、同じA、ρ、W、canonical law、順序付き制御planを構成できる。万能compilerを仮定した議論ではない。
5. stable matching、cost-awareな対選択、cache、Wの隣接相殺を追加しても、それだけでは独立機構にならない。今回それらへ案を延命しない。

既存unitary partitioning論文のtaskはVQE測定であり、その性能保証や測定節約率をPRへ移していない。還元するのはoperator構成・coherent recipeである。PR接続の実装・finite mean認証は未了だが、そのgapを新理論と呼ばない。Phase 2Cの完全なselector全体と特定論文が逐語的に同じという判定でもない。

## 7. 到達点

追加案は1件、代案は0件。今回の技術的分類は `NO_METHOD_DELTA_FOUND_WITHIN_SCOPE`。新規性を否定する一般定理は証明しておらず、この一つの具体案で独立差を擁護できなかったという限定結果である。入力不足を埋める数値探索、新fixture、F1、grant、実装・compileは開始しない。次の判断は[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)へ戻す。
