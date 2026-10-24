# R3・先行研究との差分：新規性を認定しない限定照合

前回C0、10月21/24日の索引、Track BのBM/Parent/G6対応監査を再利用した。今回は選択研究に影響する関連節だけを再確認し、全件の文献調査・新しい詳細科学レビューは行っていない。

## どの原理が既にあるか

|一次資料・使用版・位置|今回確認する対応|差分として残せない部分／不足|
|---|---|---|
|[Maxwellら 2606.30738v1](https://arxiv.org/html/2606.30738v1)、III.2.1 Eqs.10–12、III.2.3 Eq.19|symmetric BCH、nested wrapのcompact再帰、共通計算の再利用|BMの同じ三次operator/floor/internal分解は既知再帰と一致。有限signal評価や新DF boundを得たことにはならない|
|[Cugini–Atif–Subaşı 2603.13495v1](https://arxiv.org/html/2603.13495v1)、II Thm.1 Eqs.7–10、III、IV.1–IV.2|固定protocolのcost×second-moment、q比例p/sqrt(cost)、biasとZeroFill/Discardの区別|G6の固定辞書下界とParentの内側IS原理は既知の特殊化。zero-cost/full support、range/ceil、取得費用を消して最適十分Nとしない|
|[Peetz–Smart–Narang 2407.21095v2](https://arxiv.org/html/2407.21095v2)、II.3、IV.2、Note5 S13–S15|CTSの有限分解・Pauli collectionとcommon-angle構成|一般involutionの非可換wordはPauliに閉じるとは限らない。local Euler一致と全sampler/access一致を分ける。v1をv2の代用にしない|
|[Koczor 2402.15550v2](https://arxiv.org/html/2402.15550v2)、II.1–II.3、Eqs.1–6|process辞書・係数/sampling/L1最適化|operator first momentとfull channelの制約は異なるが、明示した投影/embeddingで既知線形枠組みと比較可能。単にtask名を変えた新規性にしない|
|[Casaresら 2606.30741v1](https://arxiv.org/html/2606.30741v1#S4)、IVとStep7|factorization、grouping、誤差/step費用、remainder strategyを組み合わせる設計|prefix・norm・費用・誤差を一緒に選ぶ目的自体は既知。安価な情報が何を追加するかが必要|
|[Güntherら 2503.05647v2](https://arxiv.org/pdf/2503.05647v2)、App.A3/E（C0既読範囲を再利用）|高次signed PR、RTEと資源評価|高次PRや負時間・反復負担を初提案としない。今回全本文再読ではない|
|[Poianiら 2406.03033v1](https://arxiv.org/html/2406.03033v1#S2)、§2/§3（C0の確認範囲を再利用）|cost/fidelity/bias制約の下でのbest-arm選択|少数照会・regret・abstainは一般枠組みの範囲。今回のempirical proxyにknown fidelity-bias保証はない|

SCU/SPSは初期取得後、既存Parent監査が使用したv2を本文照合に採用した。旧監査の引用版と今回の読取版の差を隠さない。Green/free-productの既知性、Wan/PTSCの詳細対応はG6_NOTE §3/§12とPA_ART/R0_AUDITの既存本文監査を継承し、この作業で全原典を再取得・再証明したとは表示しない。

## 構造ごとのR3との差分判定

|候補構造|R3を超えて得られた情報か|第2研究の独立した選択構造か|
|---|---|---|
|S2のsame-R law class|R3一般論より具体的な既存SOURCE関係。PR-2で検証済み|いいえ。再利用class/次元削減の情報だが、biasと価格の順位を残す保証がない|
|finite-Kの法・moment・allocation|実際の入力と取得経路を具体化できる|既存RTE/既存R3の特徴を再提示するだけなら不足|
|G6局所Green/P5少数群|単なる一般optimizerではない既存の生成構造|G6の成果として保持。第2研究の新しい選択則への接続は未確立、二重計上不可|
|Uや固定辞書lower boundによる棄却|特定classで使える既存保証・反例がある|真のGの安全除外をどの情報から得るかが未閉鎖。既存IS/Cauchy下界の適用だけは新規でない|
|cluster/short-circuit校正|candidate全compileを避ける条件付き既存proxy|新しいbias guaranteeがなく、一般multi-fidelityの入力とするだけならR3停止理由が残る|
|finite-bit/zero/all-trial取得費用|正しい会計・access条件として必要|正確に数えること自体を新手法としない。C1は既に古典setup/query費用の無視できなさを観測|

**Aに必要な差分は特定できなかった。** これは未知の手法が存在しないという証明ではなく、今回読んだ研究系列・SOURCE・先行資料の範囲での結果である。R3停止、BM同値性、P_U反例、C1一時凍結は既存研究の状態として保持する。

一方「全PRアルゴリズム研究がR3の重複で終了済み」も誤りである。G6の構成と条件付き数理、PR-2の限定資源実証、Parent S4の限定trade-offは別の成果として残る。Bを提案する理由はこの区別にあり、独立選択法の継続GOを意味しない。
