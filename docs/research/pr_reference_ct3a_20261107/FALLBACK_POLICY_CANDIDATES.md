# CT-3A：fallback候補と既存資産の適合性

固定参照 `6b0b6669bbf58ef22e789bfef134a1a6da9f2343`、2026-11-07 JST。[REPORT](REPORT.md)。静的な理論・SOURCE調査であり、候補の実装、word/net生成、価格評価は行っていない。

## 1. 数論的Clifford+T合成

| 項目 | 確認内容 |
|---|---|
| 既知構成 | Selingerの候補格子→norm方程式→exact synthesis、Ross–Selingerの分母順探索 |
| 停止性 | 完全な候補列挙・norm解判定を持つ数学的探索と、固定seed・有限attemptのpygridsynthを区別する。要求分母での成功、有限数値判定の保証が残る |
| strict phase | det=1のRZはstrict近似可能。projective経路や別phase fieldを無条件に文字列へ落とさない |
| T上界 | 成功した分母指数kと正規形を前提にT≤2k。kの候補存在だけでnorm解まで保証しない |
| 実装資産 | 固定pygridsynth2.0.0、repo acquire/strict_guard、保存word/guard、CT-1 emitter |
| 不足 | 固定有限epsilonの全入力で必要な解が得られる証明、数値境界処理、実際の出力wordへの一様cap |
| 判定 | 理論の条件付き価格式は`THEORETICAL_ONLY`、現backendをtotal fallbackとする資格は`UNQUALIFIED` |

[Selinger v2](https://arxiv.org/pdf/1212.6253v2) Theorem22は候補の存在、Fact27はexact合成のT数を扱う。Algorithm23と§8の効率論にはnorm解取得と素数分布の仮定が関わる。Remark26のdet条件はRZで満たす。**候補存在、norm可解、探索成功、期待効率は別。** [Ross–Selinger v3](https://arxiv.org/pdf/1403.2975v3) Proposition8.1の終了時correctness、8.2のfactoring oracle下最適性、Hypothesis8.3下の8.8/8.11の期待結果を、固定seedの有限worst-case capへ転用しない。

CT-2の端数処理を再掲する。0<eta≤1/2で

\[
C=\tfrac52+2\log_2(1+\sqrt2),\qquad
k_\eta=\lfloor C+2\log_2(1/\eta)\rfloor+1.
\]

当該kで必要なnorm解と正規形を得られる場合だけ、T≤2k_eta。etaは理論spectral精度であり、repoのFrobenius guard許容epsilon_Gとは換算・余裕が必要。この式へ保存precisionを代入して新価格を得ていない。

数学的な全探索に完全なfactorization/norm判定を組み合わせる余地はあるが、新backendの仕様・停止性・実出力価格を今回資格したことにはならない。ランダムseedingや再試行の無断採択もしない。

## 2. strict SU(2)のSolovay–Kitaev

| 項目 | 確認内容 |
|---|---|
| 既知構成 | Dawson–Nielsenのbalanced commutatorによる再帰 |
| 停止性 | inverse閉包・密な有限生成集合、全域初期net、有限精度手順が必要。深さが有限でも各数値副手順の完了は別条件 |
| strict phase | SO(3)の商空間だけでなくSU(2)の符号/位相を保持。下記のdet=1 kernelへの記号的展開が可能 |
| T上界 | ideal exact-realではc=12、eta0≤1/256を十分条件とする導出。資格済み初期最大T=t0があればT≤5^n t0 |
| 実装資産 | exact cyclotomic word matrix、guard、actual adjoint emitter。既存Qiskit SKは参考SOURCE |
| 不足 | 実word付きstrict netとcoverage証明、eta0/t0、丸めを含む再帰誤差、guardが有限回で確定する余裕 |
| 判定 | `THEORETICAL_ONLY`。c=12は今回の記号的導出であり、実装の定数として採択していない |

[Dawson–Nielsen v2](https://arxiv.org/pdf/quant-ph/0505030v2) Definition1、§3、§4.1–4.2を限定照合した。初期netとinverse閉包を前提にする。本文の近似定数や経験的基底半径/長さは有限保証の代用にしない。§3は有限精度算術を解析から除外している。

今回のCodex導出として、strict生成集合の具体的な記号表現と、Lemma1を用いたc=12の十分定数を[証明条件文書§2–3](POLICY_TERMINATION_AND_BOUNDS.md)に示す。生成word catalogueを実際に作ったものではない。初期netと丸め保証が未充足なので、明示式を新Aの実T価格とは呼ばない。

### 保存RZ資産は全SU(2)初期netを代替しない

保存recordのguard targetはRZ(theta)である。Q=iX∈SU(2)について、任意のthetaで

\[
\|iX-R_Z(\theta)\|_2=\sqrt2.
\]

実際、差Aに対しA†A=2Iとなる。したがってRZにepsilon-closeな保存word sには、逆三角不等式で `||iX-s||2≥sqrt(2)-epsilon`。**RZ-targetの個別guardだけを証拠に、小半径の全SU(2)coverageを主張できない。** これは記号的な反証で、保存word点群を数値走査した結果ではない。追加の積を作ればよいという指摘も、未生成の積のcoverage証明を与えない。

### 既存Qiskit 1.3.0のSKはそのまま証明付きfallbackにならない

既存venvの `qiskit/synthesis/discrete_basis/` にあるsolovay_kitaev.py、generate_basis_approximations.py、gate_sequence.py、commutator_decompose.pyを読取確認した。公式[1.3.0 solovay_kitaev.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/synthesis/discrete_basis/solovay_kitaev.py)も参照した。

同実装は既定でh/t/tdgのdepth10表を**実行時生成**し、SO(3)行列の数値最近傍を使う。phase情報を保持する経路もあるため、単に「phaseをすべて捨てる実装」とは判定しない。しかし、これらのSOURCEには要求するstrict全域netのcoverage certificate、eta0/t0、epsilonからの保証付き有限深さ選択、丸め誤差証明が見当たらない。既存の基底生成にはignore_phase=Trueの重複判定もあり、証明付きSU(2)表として自動流用できない。

今回はconstructorを呼んでいないので、既定netは生成していない。4 SOURCE以外の全package監査、別環境のnetの網羅探索も行っていない。「確認した再利用資産に資格済みnetがない」という限定結果である。

## 3. 完全word探索という理論的対照

これは新しい合成アルゴリズムの提案実装ではなく、密な有限gate集合の全列挙という既知の構成原理を、停止性の論点を分けるために整理したもの。

| 項目 | 確認内容 |
|---|---|
| 構成 | inverseを含む有限alphabetの全wordを長さ順に調べ、厳密な区間上界で十分に近いwordを受理する |
| 停止性 | targetが計算可能、密性、受理にstrictな余裕、wordと検証精度を公平に進める探索を前提に証明可能 |
| strict phase | exact W付き行列を比較すればprojective quotientを使わず扱える |
| T上界 | 停止後の実wordから数えられる。現在は具体的な事前一様F(epsilon)がない |
| 実装資産 | exact_word_matrix、strict_guardの演算規約は参考になる。現acquireは完全探索器ではない |
| 不足 | uniformなcover/探索長の有限証明、有限検証スケジュール、研究比較に有用なcap |
| 判定 | `THEORETICAL_ONLY`。CT-3B実装対象には設定しない |

一つのwordの境界判定を無期限に待つ逐次探索は、この停止論証を満たさない。余裕をもつwordの検証へ必ず進む公平な列挙が必要である。コンパクト性により各epsilonに有限coverが存在しても、その最長word/T数を未取得のまま「明示価格」とはしない。coverを証明付きで探索する別手順を考えられても、現資産の有限証明・定数ではなく、今回禁止されているnet構築へ進むことになる。

## 選択

候補の数学的構成可能性と、固定SOURCEで実装する準備状態を分ける。SK型の不足を最も具体化できたが、必要証拠が揃った候補はない。新規net・backend・word・数値価格を取得せず、**mandatory STOP**。次の判断はGPT／ユーザー。
