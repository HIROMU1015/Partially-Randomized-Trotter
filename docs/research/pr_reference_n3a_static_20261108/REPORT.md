# 新A N3-A：有限辞書型Clifford+T policyの静的調査

2026-11-08 JST、SERVER 2。固定参照 **`0a6ba582d246b3b724d42c8b56475c16ebef1dd9`**、専用branch `codex/pr-reference-n3a-static-20261108`。

**Stage A〜Eの静的調査を完了、mandatory STOP。** 判定は`CONSTRUCTIVE_POLICY_CONDITIONS_IDENTIFIED`。RZの加法性、正確なdyadic丸め、基本wordのstrict certificateを組み合わせることで、完成辞書から全vacuum域へ有限にwordを返すための構成条件・誤差条件・価格条件を記号的に具体化した。一般の必要十分性、新規性GO、policyの正式採択を判定したものではない。

実辞書は未生成であり、構成は`THEORETICAL_ONLY`。全supportの実装／停止／T価格／CT RPE成功は引き続き`UNQUALIFIED`。b、辞書サイズ、内部precision、T価格和／極値／期待値の数値評価は行っていない。

## 固定科学条件

Linear H4・1.00 Å・STO-3G・DF12、元Hの厳密groundを各shotで供給し、状態準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1、保存73 bits、最終native RZのstrict epsilon=10^-11。

Qiskit1.3.0/compiler A、rz/sx/x/cx、全結合・Targetなし、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、既存D融合・basis共有・scalar/event・X/Yを保持。保存199 shot/axis、energy1.5936e-4 Ha、failure.05、phase cap .4、alpha1/720と旧RPE誤差条件を変更しない。

targetはcompiler後のliteral binary64 `RZ(theta)=diag(exp(-i theta/2),exp(i theta/2))`。up_to_phase=False、W・相対位相・負角のactual adjointを保持し、phase fit、理想pi snap、2pi key同一視、合成後の新controlは認めない。

## Stage A〜Eの結論

| Stage | 確認した成立条件／不足 | 資格 |
|---|---|---|
| A：被覆 | 保存m7のLを厳密に読み、整数最近接gridと固定bit順で全域を覆える。基本指数域と使用数はbの記号式で閉じる | 条件付き構成を導出。数値b／角度辞書は未生成 |
| B：strict誤差 | 量子化のFrobenius項はsqrt2*2^(-b-2)。保存基本Frobenius上界の和と一度だけ合算し、最終epsilon内に余裕を残す | 記号的十分条件。基本wordとemitterの実資格は未取得 |
| C：停止性 | 完成・freeze済みの有限表ならonlineは有限整数演算／lookup／連結。生成時のpygridsynth停止は別 | 完成後の条件付き停止証明。実辞書未完成なので全域実装はUNQUALIFIED |
| D：T cap | 辞書経路は実基本T和で覆えるが、既存word優先ならcache側も含むmaxが必要 | N1との比較条件のみ。実T capは未評価 |
| E：再利用 | exact matrix／guard／actual adjoint emitter／保存会計を再利用できる。新規部分は丸め・連結certificate・決定的policyの小さいadapter | 適合性の静的調査。adapter未実装 |

詳細は[FINITE_DICTIONARY_CONSTRUCTION](FINITE_DICTIONARY_CONSTRUCTION.md)、[STRICT_ERROR_AND_TERMINATION](STRICT_ERROR_AND_TERMINATION.md)、[T_PRICE_FEASIBILITY](T_PRICE_FEASIBILITY.md)。

## 数学的条件と重要な注意点

未採択のbについて、Lを保存binary64上界として厳密に解釈し、

\[
N_b=\lfloor L2^b+1/2\rfloor,\quad
-b\le j\le J_b,\quad M_b=J_b+b+1
\]

を定める。J_bはN_bの整数bit lengthに対応する上位指数であり、これらの値を今回算出していない。各基本角2^jのstrict wordが完成していれば、量子化角の通常2進展開は高々M_b wordの積になる。

近似wordは可換とは限らない。負角は、各chunkを同じ順序でadjoint化するのでなく、**正側連結word全体のactual adjoint**にする。Wもその演算子の一部として保つ。

保存基本Frobenius上界g_jを使い、実emitterがwordと一致する場合の十分条件は

\[
\sqrt2\,2^{-b-2}+\sum_j g_j\le10^{-11}-\mu,\qquad\mu>0.
\]

spectral量子化誤差をそのままFrobenius予算へ足さず、逆に既存Frobenius g_jを再びsqrt2倍もしない。Qiskitの浮動global_phaseが追加誤差を生むモデルなら、その項も同じ最終予算内で資格する必要がある。今回は余裕／precisionを選択せず、数値guardも行っていない。

RZのstrict近似の存在には原理的なdet障害はないことを[Selinger v2 Remark26](https://arxiv.org/pdf/1212.6253v2)で限定確認した。ただし、word存在、基本word取得完了、guard資格、完成辞書からの返却、実T capは異なる命題。既存pygridsynth2.0.0のstrict外側探索と内側設定を一次SOURCEで確認し、CT-2／CT-3Aの全入力停止未資格を維持した。

## N1と共通policy

[N1保存RESULTS](../../../artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json)の要求値は、pointwise `Vmax < 173790`、期待 `UV < 1602930468830332421754729/9223372036854775808`。再計算していない。

既存CT-4Bの採用wordを同じliteral keyで優先する共通policyなら、N1の固定価格を維持する候補になる。このとき要求する一様capは辞書価格和だけでなく、**優先cacheの価格も覆うpolicy全体のcap**。既存wordを辞書で上書きしてから旧K／d／閾値を使う方法は適合しない。

期待capにはcanonical lawから生成されるvacuum丸めbit／cache hitの質量を覆う証拠が必要。order・product・vacuumの独立性、bit確率1/2、2保存trajectoryの無加重平均は使わない。粗い一様和が閾値を超えてもReference劣位／no-goの証拠ではない。今回価格や分布を評価していない。

## SOURCE・既存証拠の再利用

| 読取確認したSOURCE | 再利用できるもの／新たに必要なもの |
|---|---|
| [track_c_h4_synthesis](../../../src/trottertracks/track_c_h4_synthesis.py) | exact_word_matrix、strict_guard、acquireの既存基本record。cache hit即returnと事後長さ拒否をtotal性へ転用しない |
| [track_c2_b_verify](../../../src/trottertracks/track_c2_b_verify.py) | lower_nativeのwritten product逆順、W、負角actual inverse。新しい連結wordのphase適合確認は別 |
| [pr_reference_ct0](../../../src/trottertracks/pr_reference_ct0.py) | binary key／record metadata、phase域、局所誤差・RPE会計規約。composite routeを既存backend specへ偽装しない |
| [pr_reference_ct1](../../../src/trottertracks/pr_reference_ct1.py) | strict word／adjoint／emitterと局所検証pipeline。既存の無期限worker待ちをtotal取得controllerとしない |
| [pr_reference_ct4](../../../src/trottertracks/pr_reference_ct4.py) | 採用record表、実T/T†定義、signed roleと誤差の分離、有限scope |
| [pr_reference_rotation_v3](../../../src/trottertracks/pr_reference_rotation_v3.py) | dparts／saved_path_parts／rotationのroleとliteral角、有限native台帳。既存runnerは再開しない |
| [pr_reference_transfer_error](../../../src/trottertracks/pr_reference_transfer_error.py) | IEEE phase boundと旧disk単位。今回は境界を再評価しない |
| [pr_reference_interference_pilot](../../../src/trottertracks/pr_reference_interference_pilot.py) | vacuum_phase／reference_core、actual F inverse、既存joint相対位相 |

CT-4Bの既存key読取では、`0x1.0000000000000p-61`のe=-11 recordがdyadic基本角の再利用候補として見つかった。将来の指数域／内部g_jへ適合するかは未判定。他のcache全体の新coverage認定や新価格計算ではない。基本wordは既存cacheにあっても、精度ラベルだけで積のprecisionを保証しない。[NEXT_SCOPE](NEXT_SCOPE.md)に具体的keyと次の最小候補を記す。

## 固定参照・一次資料

以下の各REPORTが固定commitに存在し、本branchに継承された内容と同じことを読取確認した。SOURCEと関連条件文書を必要箇所に限定して参照し、完了済みのcompiler／context全監査は繰り返していない。

| 固定資料 | commit |
|---|---|
| [N1数値結果](../pr_reference_n1_numeric_20261108/REPORT.md) | `0a6ba582d246b3b724d42c8b56475c16ebef1dd9` |
| [N1準備](../pr_reference_n1_preparation_20261108/REPORT.md) | `bf0a7b1b5d0afb6ead3cc83ce909bbe53514b24a` |
| [CT-5B](../pr_reference_ct5b_20261107/REPORT.md) | `0dbf1fe265176677e026ea982c0bd435987f4338` |
| [CT-5A](../pr_reference_ct5a_static_20261107/REPORT.md) | `e3e852a35f24839b42a6d282083583bb611b6a9c` |
| [CT-4B](../pr_reference_ct4b_20261107/REPORT.md) | `8012c3e9547690f0d722edac9d0b9a9391ba2f8e` |
| [CT-3A](../pr_reference_ct3a_20261107/REPORT.md) | `c21a43f444f3b7936de36acd0fbd38556ae594d0` |
| [CT-2](../pr_reference_ct2_20261107/REPORT.md) | `6b0b6669bbf58ef22e789bfef134a1a6da9f2343` |
| [CT-0](../pr_reference_ct0_20261107/REPORT.md) | `f5f16fad64748236b0160284dc6c3017974b0132` |

正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAPと既読の研究概要を適用。一次資料の新規照合はSelinger v2 Remark26周辺と、固定pygridsynth commit `65f6dd06538675d63a239009574dea5c636668b1`の[gridsynth.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/gridsynth.py)、[config.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/config.py)、[domega_unitary.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/domega_unitary.py)の該当箇所のみ。数論的価格則・SKについては固定CT-2／CT-3Aの関連節を再利用し、論文全体や新アルゴリズムの独立科学レビューを行ったとはしない。

## 未資格・完了・次担当

完成辞書、実基本word／strict certificate／実T cap、優先cacheを含む価格境界、canonical期待価格は未取得／未評価。旧有限word資産のPARTIAL_COVERAGEを、新しい辞書の全域coverageと呼ばない。common frame未取得key、B′ b2 eventの到達／word、全support CT RPE成功、18round総T／T-depthもUNQUALIFIEDのまま。

今回の実施は、固定資料・JSON・SOURCEの読取、既存key／schemaの静的確認、記号的条件整理、5文書とSTOPの作成・公開のみ。**新word／角度辞書／net生成、合成、guard、科学数値再解析、T価格／期待値／極値計算、precision選択、回路build/compile、sampling/simulation、最適化、SK、18round／T-depth計算はすべて0。** 科学moduleをimport／実行していない。旧SOURCE/raw/result/grant/ledger/marker/STOP、共有設定、他ユーザー・別Trackは変更していない。

独立した静的項目を完了し、残る実資格には範囲外の実装・取得・検証が必要なため、[mandatory STOP](../../../artifacts/pr_reference_n3a_static/2026-11-08/STOP.json)。次担当は**GPT／ユーザー**：本構成条件を評価し、進める場合は[NEXT_SCOPE](NEXT_SCOPE.md)のb／内部予算／有限key／検証対象を具体化して別途承認する。CodexはN2/N3実装・CT-3B・全support合成・18round計算へ自動進行しない。
