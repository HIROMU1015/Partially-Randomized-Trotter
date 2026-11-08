# 有限precisionとrole別joint予算の候補

2026-11-08 JST。基準460726569c60ea4cf8f9839f2aff16f84157babc。[REPORT](REPORT.md)。以下の数値は保存値の引用だけで、新しい評価結果・正式採択ではない。

## 1. Precisionとword再利用

| 選択肢 | 最小変更・既存資産 | 保証と実装の条件 | Tへの作用／未確定事項 |
|---|---|---|---|
| b36固定＋role予算 | 現wordと43基本wordを保持、中央を一様capから解放する条件式 | V0-Aの下界超過を保持し、全role／emitter／shellを含むjointで閉じる。B′では旧中央safe upperだけでは不足 | Word不変なら物理Tは不変。保証成立は未確定 |
| 従来V1の有限b | 36…b_max。旧b36行は保存、37…終端を別診断。既存round_indexを再利用 | Q、G_known、missing、actual adjointを明示。現providerのB定数やdomainは変更しない | Q減少とguard／chunk増加は別。実T未評価、単調性を仮定しない |
| 保存recordの再選択 | C0 299、43 basic、85 certificateを優先。同literal・strict保証の別recordは後続catalog候補 | 固定C0優先を変更するなら別policy。旧strict inventory全体の再探索・価格rankingは今回の次診断にも含めない | 保存guardと実離散Tの比較には別数値承認。Precisionラベルだけで優劣を決めない |
| Signed digit／window | 既存positive基本wordとcomplete actual adjointを利用する別構成 | carry、指数域、word順、relative位相を新たに資格する必要。V1固定のascending binary表とは別候補 | Qは同じgridなら不変。Chunk減少がT減少になるかは価格依存 |
| 有限missing又は直接中央word取得 | Missing指数／不適格literalだけを対象候補にする。既存acquire／strict_guardを将来再利用 | Budgetとkeyと要求epsilonを先に固定し別承認。新しいepsilon ladderを自動追加しない | 取得成功・guard・Tは未取得。新合成が必要と今は断定しない |

Theta=p/2^eに対し、候補終端は`b_max=max(36,max e)`。Grid角a_b=n_b*2^-b、delta_b=|theta-a_b|、保存のsqrt2上端sを使う設計式は

`Q_b=s*delta_b/2`, `G_known,b=sum_{j in J_b intersect [-36,6]} g_j`。

既存guardを持たない選択指数はM_bとして残し、

`e_logical <= Q_b + G_known,b + sum_{j in M_b} g_j`。

Missingがなければ右辺を数値上界にできる。Missingがあれば右辺全体は未確定で、既知部分だけを完全な上界と呼ばない。`R_ax=cap_ax-Q_b-G_known,b`は指定構成のadditive証明へ残る予算である。R<=0はその証明の不足であり、真の誤差下界ではない。R>0でも、missing guardとemitter／shellの資格は残る。

Tの記号式は`T_word=sum_{j in J_b} t_j`。保存価格のない指数は未知であり、0や対数近似へ置換しない。選択chunk数だけを出すV1をT価格集計と混同しない。新表はascending binary構成の比較であって、全ての保存word再構成の最適性証明でもない。

次診断は中央4 literalだけのb変更と非中央C0／85固定を組み合わせる。Global Bを変えて非中央も再構成するpolicyではない。実装へ進む場合は同一literalへ両arm共通の有限overlay等を設計・資格し、その取得・guard・runtime・Tを別scopeで比較する。

## 2. 一様capとrole別配分

[M17_INPUTS#/condition](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)の保存値をそのまま転記する。これはkey件数や期待値ではなく、context条件C下の**物理RZ数の安全上界**である。

| Count group | B′ X（Y差分は後述） | Reference X | SOURCE上の意味 |
|---|---:|---:|---|
| D | 810025624 | 808453440 | 既存D境界／actual SOURCE inverse templateを含む |
| frames | 12884901888 | 12884901888 | 最大3qr runとcontext-max basis／inverse対 |
| rotation | 8388608 | 4194304 | B′最大2qr、Reference最大qr |
| products | 67108864 | 16777216 | K2 product/support上界と各native lowering |
| event_phase | 131072 | 0 | B′のoccurrence別P。Referenceはvacuumへ入る |
| scalar_and_wrapper | 5 | 5 | Scalar／vacuumとX/Y wrapperの混合group |
| CSWAP | 0 | 176 | B′SOURCEに不存在。Reference固定16 CSWAPのlowering |

Yではscalar_and_wrapperが両armとも6、それ以外は同じ保存値。Norm上界へ負の物理多重度を入れない。Dのsigned price補正は、literalのunionを覆うmax guardと非負D上界へ置き換える。Reference actual inverse supplementもそのunionへ含める。

同じNを維持する保存joint予算をE_arm、各axisのrole/key誤差上界をeとすると、基本の候補は

`E_ax <= sup_omega sum_{role,key} n_role,key(omega,ax) e_role,key + E_emit + E_shell`。

安全な簡約は`sum_role L_role,ax * max_key e_role,key`。異なるroleのmaxが同時に到達しなくても安全であるが、保守的になり得る。これを鋭くするには、同時到達supportの制約を資格する。期待値や有限cohort最大で置き換えない。Canonical lawの価格期待値と全support誤差supを分ける。

中央2 keyのlogical error上界をe0,e2とすれば、既存safe-max条件はB′で`2qr max(e0,e2)`、Referenceで`qr max(e0,e2)`。各trajectoryのorder数k0,k2を使う候補では`k0+k2=qr`を保持する。全supportで最悪orderが全位置に可能なら、単なるorder別精度化からmaxより小さい一様保証は得られない。Reachabilityを確認せずその極値が実到達するとも断定しない。

既存controlled-RZの±半角を同一word／actual adjointで実装する局所pairや、basis共役blockの直接joint評価は、三角上界を改善する別候補になり得る。ただし、理想SOURCE inverseと近似wordのactual adjointを同一視せず、word選択・native順序・control条件を確認する必要がある。既存C2の共役block上界を自動移送しない。Pairの符号が逆という理由だけでrelative角の誤差や量子化を取り消さない。

### 保存資産で先に調べられるrole

次診断は、C0の保存`strict_error_upper`と85 certificateの`quantization_upper + word_error_upper`で、D／frame／product／CSWAPの有限key maxを作る。85のemitter項はlogical和へ重複投入せず、別の残項へ帰属する。C0／85を再合成・再guardせず、各roleのm17移送は[条件CとSOURCE閉包](../pr_reference_n5b2_static_20261108/JOINT_SCALAR_FULL_PATH_CONDITIONS.md)付きとする。

Scalar_and_wrapperは混合groupのため、wrapperだけの既知guardでgroup全体を覆わない。Event・vacuum・域外scalar・coverage不明のroleはunknown。既存の共通joint scalar因子分離は、同一logical word列、同一named命令・wire・consumer、terminal sink条件の下の信号不変性である。これを採用するか未決のため、emitter／shellを0にせず別変数で残す。旧certificateも変更しない。

新しい予算式が成立した後の接続は、旧d_oldとsynthesis diskを同じraw単位で一度だけ合算し、同じrho0／PF控除／alphaで保存N十分性を確認する手順である。保守的なX/Y共通joint上界に対する旧変換は2sqrt2倍。Axis別の楕円／鋭いgeometryを新たに採択したり、PFや旧phase diskを二度足したりしない。今回RPE式は呼んでいない。

## 3. Domainと位相の残項

| Role | 保存された問題 | 今回の扱い／将来候補 |
|---|---|---|
| B′ scalar | signed `-0x1.86e032ee41d40p+12`は実SOURCE literalが現域外 | 現providerはlookup前reject。別domain構成が必要。Random非到達で除外しない |
| Reference vacuum | 保存包絡32751.887444214277 radが現L=82.49247654978845を超える。全key列挙なし | 包絡超過は実到達角の域外証明ではない。広いdomain又は強い到達域証拠という分岐を保持 |
| B′ event r32 | 96 positiveの保存安全superset、うち70が現域外 | Supersetをcanonical到達集合と同一視しない。未観測を確率0としない |
| Reference event | 独立controlled event P roleは0 | Vacuum算術への寄与を消去しない |

既存j6反復D1、有限指数拡張D2、strict符号／relative角差を保持する周期表現D3は[旧候補](../pr_reference_post_n5b3_policy_20261108/POLICY_OPTIONS.md#4-domainを直す候補)として残す。今回の有限中央表へdomain数値診断を追加しない。ProviderのL／B定数変更、SOURCE vacuum生成、2pi key同一視、phase fitは行わない。

## 4. 強baseline・費用・判断

両armに同じrecord pool・b候補・guard精度・domain構成・support情報・価格取得機会を与える。診断にarmラベルを付けることは、本番providerへarm依存word分岐を追加することではない。同じliteralへ共通policyを用い、role別予算を採用するときは両armの実行可能性を確認する。

M17でB′626／Reference630を保持し、全18roundは保存各N_mと各roundのPF／r／bitsを使う。共通frame価格を総shot差から消さない。LP下界は実行可能なbaselineではなく、未資格roleの価格・誤差は0ではない。総Tが悪化する精度案を局所Q改善だけで採用しない。

次の数値診断は「既存43の細分化で何が再利用できるか」「他roleの保存logical上界がどれだけ残るか」を判別する材料である。実行可能な完成policy、一般的な最適性、全18round優越、新規性の認定ではない。**Mandatory STOP。**
