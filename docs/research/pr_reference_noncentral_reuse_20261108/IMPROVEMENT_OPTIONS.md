# 構造的誤差評価・保存word再選択の候補と条件付き証明

2026-11-08 JST。基準800554fbb64c21e584bb29a1a8f52490ab4163e0。[REPORT](REPORT.md)。以下は静的な数学的十分式とSOURCE適合条件であり、未実施の数値評価・m17成功・保証の正式採択ではない。

## 1. Template別の重み付き上界：静的に証明できる部分

各physical native RZ key kに既存logical errorの非負上界e_kを与える。1-qubit Frobenius上界はspectral上界としてembedし、joint Frobeniusへdimension係数なしに転用しない。保存template tのcontext cにおける物理多重度をn_t,c,kとする。

`a_t = max_c sum_k n_t,c,k e_k`。

Context集合が全supportを覆い、literalが同じなら、unitary telescopingよりtemplate誤差はa_t以下。旧e_maxが対象全keyを覆う場合、各cについて

`sum_k n_t,c,k e_k <= (sum_k n_t,c,k) e_max`

なので、`a_t <= max_c RZ_count(t,c) * e_max`。これを非負のSOURCE template多重度へ加算すれば、対応する旧一様max上界より悪化しない。**保証の非悪化性は証明できるが、strictな改善率や予算内への到達は数値未評価。** Guard coverageが違う、又は未知値を0にした比較ではこの結論を使えない。

RTE basis IDをbとし、帰り側もその保存loweringを使って

`f_b = max_c sum_k n_forward,b,c,k e_k + max_c sum_k n_inverse,b,c,k e_k`

と置く。前後で同じcontextを仮定する必要はない。K2では1 event最大3 application、1 applicationが新runを最大1つ開くので、既存basis共有を保った安全上界は

`E_frames <= 3 q r max_{b in SOURCE_RTE_registry} f_b`。

既存catalog全`basis:`集合はSOURCE registryを含み、D0…D3等も持つ。従ってSOURCE ID集合へmaxを制限すること自体は非悪化である。Registryのsource/hash対応、全componentからの包含、compiler context閉包が条件。有限cohortに現れたIDだけへ制限する方法ではない。Identityによるflush、occurrence終端を含め、既に共有で消えたbasis pairを再度削減に数えない。

Dは既存融合後SOURCEの非負template係数を使う。例えばB′は

`E_D <= a_left_all + a_right_all + (q-1)(a_left_no0+a_right_no0+a_boundary_control)`。

Referenceはn=q/2に対し

`lane(n,a)=2a_0half+(n-1)a_0full+2n sum_{j=1..3} a_jhalf`,

`E_D <= lane(n,forward)+lane(n,actual_SOURCE_inverse)`。

ProductsはB′`4qr a_cz`、Reference`4qr a_z`、Reference CSWAPは`16 a_cswap`を保持する。これは旧[worst_counts](../../../src/trottertracks/pr_reference_ct0.py)の非中央式であり、新しいfusion・PF scheduleではない。Dのsigned price補正を負のnormにせず、actual inverse supplementをforward native inverseへ置換しない。

## 2. 共役blockの条件付き保証とinverse defect

すべてspectral normとする。Ideal U=B D B†、approximate V=A D_tilde CでB,D,A,D_tilde,Cがunitaryとする。定義は

`eps_f=||A-B||`, `eps_i=||C-B†||`, `eps_D=||D_tilde-D||`, `eta=||C-A†||`。

D=exp(-itH)、H Hermitian、W=lambda_max(H)-lambda_min(H)なら、

`||V-U|| <= eps_D + min(eps_f+eps_i, min(2,|t|W)*eps_f + eta)`。

**証明。** 通常のtelescopingは第一候補eps_D+eps_f+eps_iを与える。CをA†へ置き換える差はeta以下、D_tildeをDへ置き換える差はeps_D以下。残る共役差は、S=A†Bとして

`||A D A†-B D B†||=||[D,S]||`。

`||S-I||=eps_f`であり、`[D,S]=D(S-I)-(S-I)D`から2eps_f以下。Duhamel積分から`||[exp(-itH),S]|| <= |t| ||[H,S]||`。Hをspectrum中心でshiftして`||[H,S]|| <= W ||S-I||`なので第二候補を得る。以上を小さい方で囲う。

Cが**同じ近似basis全体のactual adjoint**ならeta=0となる。独立の帰りwordならetaは残す。SOURCE上のinverse対だけからeta=0とは言えない。eta未取得時の安全な一般上界はeps_i+eps_fであり、これだけで構造的利益があるとは主張しない。

Ordinary controlled Dでも、B/Aが同じuncontrolled system-basisとして両branchへ作用するなら、control0は共役でidentity、control1へ上の証明を適用できる。そうした構造がなければfull joint generatorの幅で評価する。SOURCEのscalar／event／vacuum relative phaseをcommon global phaseとして消さない。Exact Wとcomplete adjointも保持する。

### 現SOURCEへの適合と限界

[DF RTE _append_basis](../../../src/trotterlib/df_rte_qiskit.py)と[DF partial-S2 _append_block](../../../src/trotterlib/df_partial_s2.py)はSOURCEでは順序反転＋gate.inverseを使う。一方、compiler Aはjoint全体を標準同値則でlowerし、[consume_native](../../../src/trottertracks/pr_reference_common_word_consumer.py)は各native RZを独立に置換する。Negative RZのwhole adjointは保証するが、二つのbasis lowering全体を同じ近似basisのadjointへ結びつけるinterfaceはない。

したがって現構成では、(i)既存word列でeta=0をsymbolic gate/token恒等式から示す、(ii)etaを別のstrict評価で囲う、(iii)近似basis全体を再利用する別consumer構成を資格する、の分岐がある。(iii)は既存compiler／consumer経路の変更なので今回実装・採択しない。Strict W／phase、wire、順序、全contextの条件が必要。C2の抽象不等式は参考にできるが、Full Y4の数値weightやwhole保証を転用しない。

RTEのrunはproductsを含み得る。D_runが小時間のrotationだけとは限らないため、全runへmin(2,|t|W)の小さい値を仮定しない。ProductsのPauli patternがidentityとなるSOURCE条件を証明する場合も、actual native binary literalへの旧compiler diskを別に保持する。定数piの近傍をexact Cliffordへsnapしない。

## 3. 反復・support・joint構造でしてよいこと

| 構造 | 正当化できる条件 | 無条件にしてはいけないこと |
|---|---|---|
| 同一keyの反復 | Context／位置別e_kを非負加算。既知のcommutator等の別保証があれば使う | 反復がrandomだからsqrt回数になるとする |
| Adjacent inverse pair | 同じwire・同じ完全word・逆順actual adjointで間に妨げがない | 符号反対だけで別word／別loweringまで相殺する |
| Basis共有 | 現SOURCE active_key規則とidentity flushを使う | 既存で消したpairを二重に削減する |
| Support制約 | 全component／orderを覆う有限alphabet、又は安全なsuperset上のmax-plus DP | 有限例の最大や期待run数を全support上界にする |
| D境界融合 | 現left/right/boundary／actual inverse構成の非負係数を使う | 新fusionやfull-Y4のblock係数へ変更する |
| CSWAP2列 | 間にF/G・vacuumがあるjoint順序を保つ | Involutionというだけで両端列の誤差を消す |

例えば同一keyの系統的な角度ずれdeltaがN回続くRZでは、spectral差は`2|sin(N delta/4)|`となり得る。小さいずれがcoherentに蓄積する反例であり、平均ゼロやsqrt Nを仮定する根拠はない。これは人工的な記号例で、今回のword誤差が角度ずれだけというモデルではない。

全support run上界をさらに鋭くするなら、active basis ID／identity flush／order0・2／終了状態を持つ有限stateのmax-plus再帰を設計できる。有限superset上の最大なら安全で、canonical確率値は不要。ただし入力alphabet・逆側literal・contextと物理多重度の対応を資格する必要がある。今回DPを実装・数値評価していない。

## 4. 保存word再選択と総T

Static metadataでは多数の同target・別sequence recordがあると確認できた。各候補は `(archive,member,spec,sequence_hash,saved_guard,stored_T)` で区別する。Record keyだけで異なる保存版を潰さず、method欠損は未資格。Negativeは採用positive wordのcomplete actual adjointを使い、別合成しない。Total T+T†はadjointで不変だが、strict Wは保持する。

次の有限診断は各literalの実保存guard／離散Tの非劣候補を残すだけとし、C0優先を自動変更しない。C0とcomposite85のlogical誤差の意味を揃え、emitter／shellを別帰属とする。少ないTと小さいguardの両候補を残し、precisionラベルだけで選ばない。全体の最適性・baseline価格はPareto表だけでは得られない。

取得済みwordで数値保証が閉じる場合は再合成を避ける。閉じない場合も、現在の加法certificateの不足、inverse defect、domain、guardの保守性を分ける。直接joint guardで済むか、有限新wordが必要かは未決。新規合成を必須とも不要とも認定しない。

両armへ同じrecord pool・構造保証・有限選択機会を与える。M17は`626*(E[T_BX]+E[T_BY])`対`630*(E[T_RX]+E[T_RY])`であり、共通frameの絶対価格がshot差へ残る。Local guardが小さくても総Tが増えるなら無条件採択しない。全18roundは各保存N・PF／r／bitsの成功条件を閉じてから評価し、LP下界を実行可能なB′費用にしない。

**残項：中央／B′scalar域外／Reference vacuum／event到達性／emitter・shell・terminalを未資格として保持する。数学的な条件付き十分式と、本番でその前提を満たした保証は別である。**
