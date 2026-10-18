# 方法差検証：既知class式へのDF代入で同じ処理になる

[判定・入口](README.md)／[機械可読判定](technical_verdict.json)／[証拠・読取範囲](evidence_index.json)／[引継ぎ](handoff.md)。以下の式変形は既知のCAR・共分散・有限tensor和の適用を対応させた技術比較である。新しい数値計算、分散定理、保証、SOURCE実装を作成・実行していない。

一次資料は [Koizumiほか2606.30601v2 S8](https://arxiv.org/html/2606.30601v2#A8) と [著者SOURCE固定版](https://github.com/koizumiyuki/reprod_ORshadow_covariances/blob/206aa441d6a08c1984975fddf14a23ccf054f370/fs_covariance.py)、[同README](https://github.com/koizumiyuki/reprod_ORshadow_covariances/blob/206aa441d6a08c1984975fddf14a23ccf054f370/README.md)。以下の「対応」「K*」「費用比較」は、これらの既知式・処理を指定比較条件へ適用した本作業の技術的推論である。

## 1. 比較する方法と適用範囲

候補は、固定粒子数η・n modesの同じstate ρ、同じorbital-rotation ensemble、同じ1-/2-RDM estimatorから、既知DF Hamiltonianの全energy分散を計算するものとする。候補の公表済み記述には、新しい測定、推定量、相関取得oracle、独立した縮約規則はない。改善として挙げられた操作は、DF因子保持、charge/incidence分類、fermion反対称性・符号・相殺保持、展開前縮約、cache/streamingである。

対照 **K\*** は、S8の既知式と著者SOURCEの係数生成規則を使い、これらの操作を全て許した具体的手順とする。対照に巨大な全共分散行列を保存させない。候補の未実装generatorをblack boxとして呼ぶ対照でもない。

S8全表・著者READMEが示す十分な範囲は `n>=8`、`4<=η<=n−4`。一般のRDM estimatorの定義域とこの共分散表の非退化条件は区別する。端のfillingで分母を勝手に正則化せず、今回その拡張は検討しない。近似stateのenergyをground energyとみなす仮定も追加しない。

## 2. Hamiltonian・添字・符号の対応

### DFからのnormal-order

\[
H=cI+Q+\sum_{\ell=1}^{R}w_\ell A_\ell^2,
\qquad A_\ell=\sum_{pq}(L_\ell)_{pq}a_p^\dagger a_q
\]

と書き、ここではHermitianなL、一つの実weightを各squareへ吸収する規約を用いる。既存入力の1/2等はwに含める。違うDF規約を同じ係数と誤認しない。CARによる通常のnormal-orderは

\[
a_p^\dagger a_q a_r^\dagger a_s
=\delta_{qr}a_p^\dagger a_s+a_p^\dagger a_r^\dagger a_s a_q.
\]

従って一体係数は `h_eff=h+Σ_l w_l L_l²`。増加順pair `p<r, q<s` に対する二体係数は

\[
G_{(p,r),(q,s)}
=2\sum_\ell w_\ell\big[(L_\ell)_{pq}(L_\ell)_{rs}
                         -(L_\ell)_{ps}(L_\ell)_{rq}\big].
\]

これは全順序indexの和をcanonical pairへ集めた通常のantisymmetrizationであり、新しい係数生成則ではない。factorを記号のまま保持でき、Gを全展開する必要はない。Qが既にnormal-order補正を含む別規約なら、元square表示と照合し二重加算を避ける。保存DF入力への代入・数値生成はしていない。

scalar cは絶対energyへ戻す。`Var(c+Z)=Var(Z)`により分散計算から外すだけで、Hの基準原点を変更しない。DF近似誤差や状態品質の問題を解決したとはしない。

### 紙面とSOURCEのrow/column規約

論文S1.2、(S17)–(S20)は
\(D^P_{\rho;Q}=\operatorname{Tr}(\rho a_P^\dagger a_{Q,\rm reverse})\)
を採用し、matrix elementはrow Q・column P。traceは `binom(η,k)` であり、trace1へ再正規化しない。

著者SOURCEの `rdm_value(psi,A,B)` は**Bをcreate、Aをannihilate**する。`estimator_poly(left,right)` はRのrow=left・column=rightのminorを使う。従って紙面の `(P,Q;R,S)` と対応するSOURCE呼出しのtupleは **`(Q,P;S,R)`**。係数側も同じ変換をする。estimatorの参照は論文(S146)–(S147)、trace規約は(S21)である。

この対応でSOURCEの `charge_profile` の符号規約が紙面(S744)と一致する。関数引数の名前だけを同じにして両者を照合するとchargeが反転する。複素共役、canonical pairへのsortとwedge signも明示的に対応させる。これは静的な規約照合で、コードを数値テストしてcorrectnessを認証したものではない。

## 3. 既知の共分散kernelと必要なstate情報

S8の主な対応は以下の通りである。詳細な式番号・関数位置は[索引](evidence_index.json)に固定した。

| 紙面 | SOURCEの処理 | K\*での用途 |
|---|---|---|
| S1のunnormalized RDM、S3のestimator | `estimator_poly`、`rdm_value` | 一体・二体の同じ推定量と真の平均 |
| (S742)、(S759)–(S760) | `sector_coefficient_cached`のestimator積 | 1–1、1–2、2–2とHermitian共役のmixed momentを含める |
| (S744)–(S758) | `charge_profile`、`make_cases` | chargeとincidenceで係数classを共有。structural profileだけでは不足 |
| (S756)、(S763)–(S765)、Alg.3 | `trace_factor_poly`、`prefactor`、`sector_coefficient_cached` | occupation/hole blockの係数A_W |
| (S766)–(S769)、Alg.4 | `merge_sign`、`coefficients_cached` | hole展開の包含排除からΓ_K、sort時のfermion符号を保持 |
| (S769)とmixed版 | `covariance_from_gamma` | 必要RDMとの線形結合と平均積の差引き |

行列Cの成分を、同じ複素共役covariance規約で

\[
C_{\mu\nu}=\mathbb E[\hat d_\mu\overline{\hat d_\nu}]
                  -d_\mu\overline{d_\nu}
=\sum_{K\subseteq L_0}s_{\mu\nu K}\Gamma_{\mu\nu K}
      D^{(t)}_{\rho;I_K,J_K}-d_\mu\overline{d_\nu},
\quad t=\ell_++|K|
\]

と略記する。Γとsは既知係数とmerge signであり、新規に導出したものではない。

重要な実装上の区別は、**impossible chargeが消すのはraw second momentであって、covariance全体ではない**こと。(S755)の後の説明と `covariance_from_gamma` は、その場合にも平均積の負号を残す。K\*はraw second momentを集計し最後に全平均の絶対値二乗を差し引くこともでき、この区別を保てる。

| estimatorの組 | 一般に現れ得るRDMの最大order（S8既知展開） |
|---|---|
| 1–1 | 2 |
| 1–2、2–1 | 3 |
| 2–2 | 4 |

particle numberにより存在しないorderは除く。全pairで4-RDM全体が必須という下界ではない。係数を集めて特定の線形結合だけを要求すること、symmetryで消すことは両法で許す。しかしDFがあることだけで1-/2-RDMへ閉じる、SlaterのWick則を一般相関stateへ使える、という仮定は置かない。

SOURCEの `expectation_symbols` はpure stateのoccupation振幅辞書を走査する。密なsector振幅ならその保存量は `Θ(binom(n,η))`、固定orderの1つの相関評価もその走査費が必要になる。実験から相関を得る場合にはこのoracle自体の取得誤差・copies・古典処理が必要で、無料のO(1)入力ではない。quantum `Var_ρ(H)` と、測定estimatorの分散Cも別であり、ground stateだから後者も0とはしない。

## 4. K\*の具体的な演算手順

以下は紙上手順であり、実行可能SOURCEを新規作成したものではない。

1. **入力をそのまま共有**：n/η、c/h、w/L、mode順、precision、ρ相関の供給方法とcache規則を共通にする。
2. **既知のnormal-order**：§2に従いh_effを計算する記述を置く。二体係数はwとL二枚の積・exchange項のfactor listで表し、Gを全保存しない。
3. **既知のclass係数を共有**：1–1、mixed、2–2のcharge/incidence representativeごとにS8のA_W/Γ_Kを用意する。relabel・Hermitian共役・sort signを戻す。全absolute-index pairごとのcacheは強制しない。
4. **全energyの有限和へ代入**：係数をvとして `Σ_{μν} v_μ conjugate(v_ν) C_{μν}` を作る。一体×二体の両向きと二体block ℓ×mのcross項を落とさない。cは平均にだけ加える。
5. **標準のfactor contraction**：同じclassの項をまとめ、Γ・charge/incidenceのindex制約・符号・必要RDMと、h_effまたはw/L因子を一つのtensor和として縮約する。rank/index loopを交換し、適切な順序で中間tensorを作り、使い終われば破棄する。signed sumを保ち、項別絶対値boundへ変更しない。
6. **state取得も同じ順序で共有**：RDM成分または係数をまとめた線形結合の取得、cache、streamingを候補と同じ情報条件で行う。高次相関の未知性を一方だけ免除しない。
7. **平均積を差引く**：raw momentを先に集計した場合は全非定数energy平均の絶対値二乗を引く。Hermiticityに従って実部を取る条件、算術precision・丸め順を両側で揃える。

これは**既知S8式に対する係数代入と通常の縮約順序選択**である。候補が挙げた相殺・DF因子・class・storageの工夫は全てこの手順内で選べる。新しいkernel、独立したstate情報、非標準の縮約恒等式を候補側だけに供給していない。

## 5. 演算量・peak memory・保存量の比較

### 具体的に実現可能なstreaming上界

nをmode数、RをDF block数、bを算術bit/precision条件とする。dense Lを入力に持つ場合、通常のh_eff構成は `O(R n³)` 個のdense算術演算、因子入力は `O(R n²)` storage。canonical二体係数を必要時だけ計算する費用は1成分あたり `O(R)`。これらは候補・K\*共通で、実測時間や最良計算量ではない。

全1-/2-body係数数をJ=`n²+binom(n,2)²=O(n⁴)`とする。**単純な一つの実現例**はJ² pairをstreamし、class係数と高々固定数のRDM項を評価する方法である。算術上界は `O(R n⁸)` に共通前処理・相関取得・Γの生成費を加えたものとなる。全共分散 `O(n⁸)` storageは不要で、factor、h_eff、class係数と限定cache、1pair分のworkspaceで足りる。この例を最適化済み対照の性能と呼んだり、候補のspeedupの分母にしたりしない。

### 展開前factor contractionも両側で同じ

本比較で実際に使うK\*は、上記の遅いpair列挙に固定せず、§4.5のfactorized tensor networkを使う。1–1はh_eff二枚、mixedはh_effとw/L/L、2–2はw/L/Lを二組とし、既知Γ・符号・index制約・相関tensorを接続する。antisymmetrizationは有限個の符号付き項であり、同じ順序で合算できる。

縮約順序πの各段階sにおけるindex集合をU_s、各indexのdimensionをd_i（modeはn、blockはR等）、その時点のlive tensor集合をLive_sと書く。通常のdense二項縮約の紙上費用記述は

\[
F(\pi)=\sum_s O\!\left(\prod_{i\in U_s}d_i\right),\qquad
M(\pi)=\max_s\sum_{X\in\mathrm{Live}_s}\prod_{i\in\mathrm{indices}(X)}d_i.
\]

streaming・sparsityを用いるなら同じloop範囲とcacheを双方へ与える。index制約・sort/sign判定の生成費、denseな制約tensorを材料化する場合の費用も含め、無料の巨大maskを入力として与えない。bit演算費、Γ生成費、相関取得費T_ρ、相関取得workspace M_ρも別に加算する。Fをwall秒と同一視しない。

| 費用・情報 | K\* | 現候補を許された操作で具体化したもの | 残る独立差 |
|---|---|---|---|
| 共通normal-order/DF入力 | 同じh_eff、w/Lと符号付きexchange | 同じ | なし |
| class生成・係数精度 | 同じS8 representative、Γとsort/共役 | 同じ | なし |
| 縮約算術 | 同じfactor listに同じπを選択、F(π) | 同じF(π) | なし |
| peak memory | 同じlive tensor・stream/cache条件、M(π)とM_ρ | 同じ | なし |
| 保存量 | 因子・必要class・同じ中間/相関cacheのみ。全Cを保存しない | 同じ | なし |
| state情報 | 同じRDM成分/線形結合、同じ取得回数・precision・T_ρ | 同じ | なし |

独立差を消したのは「最強対照もいつか速くなるかもしれない」という期待ではない。**K\*の既知kernelと明示したfactor listが同一なので、候補が選ぶ任意の通常のπ・streaming・cacheをK\*でも同じ演算列として選べる**という対応である。候補専用generatorを呼ばずにこの対応を与えた。実装にないHamiltonian loopを追加しても、既知kernelの通常の応用・実装最適化という分類は変わらない。

候補にπ自体が未提示なので、独立した改善exponent、定数倍、最適memory frontierの数値を捏造しない。新しい非標準縮約則・相関取得法・一様分散保証が将来提示された場合は本比較の外だが、今回はそれを探索しない。

## 6. 著者SOURCEをそのまま実行する場合との区別

固定SOURCEはsymbolic class係数と確認用state評価を実装する。`@cache`は具体的tuple等をkeyにし、全分子のDF全energy loop、rank依存の最適縮約器、相関取得費込みのbenchmarkを提供するわけではない。`covariance_from_gamma`には係数の`evalf(50)`等もあり、プラットフォーム数値certificateとは呼べない。

READMEの「64 listed representatives」、紙面Tables 2–4の細分類、v1を記すmodule docstringとv2へリンクするREADMEは、そのまま区別して記録する。全class間の機械検証された一対一対応、係数の完全性・正しさ、全端点fillingを今回認証しない。これらの未検証事項を、新しい候補の独立差と取り違えない。方法差の還元は、既知S8の式と与えられた最適化許可に基づくものであり、SOURCEの既存CLIが候補をそのまま実行できるという主張ではない。

**結論は指定された候補の `REDUCES_TO_KNOWN_METHOD`。** 実装最適化の有用性一般・全fermionic測定の最適性・新規性/論文化・科学的GO/NO-GOを決めず、この候補の探索はここで止める。
