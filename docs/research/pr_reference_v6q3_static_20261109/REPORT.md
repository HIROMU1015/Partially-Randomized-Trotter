# 新A V6-Q3-S：全support受入れ・normal域の静的調査

2026-11-09、SERVER 2。**全support受入れは未資格のまま。** exact入力・半角・集約・返却値のRN検査はSOURCEに対応する。normal域の証明を有限generatorの監査へ縮約できる条件を整理したが、実科学literalの格子幅を今回は評価していない。m17 schedule callerへの接続も未実装である。新しい科学計算・回路・再会計は実行していない。

[証明義務とSOURCE対応](ALL_SUPPORT_ACCEPTANCE.md)／[必要な場合の最小Q3-N計画](NEXT_VALIDATION_PLAN.md)。以下の静的論証は著者自身の技術調査であり、独立科学レビューや新保証の正式採択ではない。

## 1. 固定資料と範囲

| 固定資料 | 今回の使い方 |
|---|---|
| [Q2結果 9560e310](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9560e310e9f848fe59a1dafce4b506d5a08aae76/docs/research/pr_reference_v6q_q2_20261109/REPORT.md) | 保存4構成の有限correctness結果・CHECKS・STOPを引用。再実行しない |
| [Q1 provider eaf70d88](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/src/trottertracks/pr_reference_v6q.py)／[既存証明整理](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/docs/research/pr_reference_v6q_preparation_20261109/B1_IMPLEMENTATION_AND_PROOF.md) | 実装不変更。未解決条件を今回詳しく分解 |
| [V6-P結果 5f672ce6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md) | M・B_norm・条件付き結果を引用するだけで再計算しない |
| [元科学SOURCE・入力 f2638a78](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f2638a7845685a119619382b850441f5d8d33713) | D/RTE/basis、F/G assembly、旧phase_bound、compiler Aの正本 |
| [V6設計 e5d6e388](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5d6e388ee2543b47ebcc1dfd9e19862d08ac327/docs/research/pr_reference_v6_static_20261108/NEXT_VALIDATION_PLAN.md) | B1の意味・旧誤差を残す規約を継承 |

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit 1.3.0／compiler A／rz,sx,x,cxを固定。B′の4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar／signed event・X/Yも変更しない。energy精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis 1/720を維持する。

対象は保存m17：q=131072、r=32、bits=88。保存Reference 630 shots/axis、条件付きB1 626 shots/axisは既存結果として保持し、今回算出し直さない。m0〜m16はlegacy provider／旧保証のまま。新A以外のTrackは対象外である。

## 2. 独立した判定

| 証明義務 | 判定 | 今回確定したこと／不足 |
|---|---|---|
| 入力・half・和・合計一回RN | `PROVED_STATIC` | 受入れ型の格納値をratio→Fractionへ移し、exact半角・両lane和の後で唯一のfloat変換。返却値はexact中点でRN-even検査。runtime前提を明示 |
| 全supportのgate family・shape・順序 | `CONDITIONAL` | 既存A1は全D／全registry basisを対象としてRZ／XX+YY族を記録。builder文法はこの有限poolから族を増やさない。真正gate・固定poolの型／有限性条件は別途必要 |
| 全supportのliteral型・finite値の受入れ | `UNQUALIFIED` | numeric生成経路は追跡できたが、固定pool全体の受入れmetadataをQ2の4例から外挿しない。有限監査に縮約可能 |
| exact Sのnormal域 | `UNQUALIFIED` | 共通格子・半角・正規化を含む十分条件を導出。実際の格子幅ℓ≤1022は未確認。固定条件内の反例も未取得 |
| 保存Mからの上側範囲 | `CONDITIONAL` | 同じ旧prefix certificateが適用されれば各lane≤M、合計≤2M。旧数学的相殺を再度改善へ計上しない |
| 保存B_normの継承 | `CONDITIONAL` | SOURCE生成・正規化・controlled対応を変えず、providerは読取だけ。上流normalization項を保持。旧算術モデルの条件を引き継ぐ |
| m17限定selector | `PROVED_STATIC` | exact modeはbuiltin整数m=17だけ。defaultはlegacy。q/r/bitsを検証する機能ではない |
| 実schedule caller | `UNQUALIFIED` | 旧assembleは旧phase和を返す。実m17のq/r/bitsとのbindingは未実装 |
| relative Pと信号 | `CONDITIONAL` | genuine固定族・vacuum ray・SOURCE inverseの条件下で既存coherence式を維持。Q2のnegative witnessは有限根拠として継承 |
| native RZ数の不変性 | `CONDITIONAL` | 同一contextでP角だけを変える構造的補題は再利用可能。実m17 consumerへの接続は未資格。新価格を取得していない |

これらを合算した全support PASSは宣言しない。型、normal下界、算術モデル、caller、価格接続はそれぞれ別の義務である。

## 3. normal域を閉じるために具体化できたこと

**basisの対をexact和から除ける。** Dは同じ記録basisのinverse→diagonal→forward、RTEは同じbasisのinverseで入りforwardで閉じる。basis共有は対をまとめ、最後のactive basisも閉じる。固定RZのinverseはbinary角の符号反転で、XX+YYは真空上の位相0を保つ。したがって、完全な各block/runのbasis RZの寄与はFractionの和で厳密に0となる。

これはbasis回路を削除・融合する提案でも、旧naive加算の丸め誤差が消える主張でもない。旧Mもこの数学的相殺を既に利用する。今回の利点は、**normal下界用の格子監査からbasis角の指数を除ける**こと。providerは全parameterを読むので、basisの型・有限性監査は依然必要である。

残る有限generatorは、Dの完成済みbinary primitive（4 half blockとblock0 full）、固定r32の二つの中央角、scalar、0／π／数値周期τである。global_phase生成の加減算と正規化が共通格子を保つ算術モデルなら、任意のevent列のg_F/g_Gも同じ格子に入る。角のexact半分を含めた幅が2^-ℓ、ℓ≤1022なら、S≠0のとき|S|≥2^-ℓ≥2^-1022を得る。trajectory列挙やm17 whole buildは不要である。

ただし、binary64一般の格子だけでは半角後に2^-1075まで細かくなり得て、normal下界を満たさない。また、旧剰余誤差のγ上界だけから格子の閉包は導けない。正しく丸められた剰余という旧SOURCEのモデルを使う条件を明示する。今回、実入力の指数・ℓ・min/max・位相を新たに計算していない。

## 4. M・B_norm・費用へ接続できる範囲

保存M=16375.943495244212、B_norm=5.172131038537467e-08はV6-Pからの引用。新providerは入力literalを読み、取得済みgの上流誤差を変更しない。同じcertificate、normal域、runtimeモデルの下で

\[
|S_F+S_G|\le 2M,\qquad
|\Phi-S_F-S_G|\le 2uM,\qquad
\delta_{B1}\le B_{\rm norm}+2uM,\quad u=2^{-53}.
\]

Qiskitのfloat周期正規化、scalar／signed event、controlled SOURCEとの対応を0誤差へ置き換えない。相対Pをcommon joint scalarとして除去しない。固定compilerのP→1 RZは角を保つ理想native恒等式（共通joint phaseを除く）であり、Clifford+T価格不変の主張ではない。

V6-Pの630→626および条件付き全18round総RZ利益は、引き続きその条件を伴う既存結果である。全support受入れ・実callerが未資格なので実装済みの全18round優越へ昇格しない。総T、T-depth、classical走査費、CT domain／emitter／wordの旧課題も解消していない。

## 5. 実施内容・保全・STOP

今回実施したのは固定SOURCE／一次Qiskit SOURCEと既存報告の読取、数理的条件整理、3文書の作成・公開・remote照合だけ。新しい補助実装や人工テストは追加していない。既存Q1テスト・Q2・科学runnerは再実行していない。科学cacheのunpickle、保存literalの指数評価、位相計算、回路build／compile／simulation、sampling、guard／合成、新価格、RPE／RZ／T再会計はすべて0件である。

文書の技術確認では、3文書の相互リンクと固定project SOURCEリンク19件のGit object存在を確認し、差分の空白検査を実施した。科学コードをimportせず、文書参照の確認に限定した。

親は9560e310、専用branch `codex/pr-reference-v6q3-static-20261109`。変更対象は本ディレクトリの3文書のみ。旧SOURCE・raw・結果・grant・ledger・marker・STOPと他Trackは不変更。共有設定や他利用者のprocess／fileを変更せず、AI subagentは使用していない。

**公開・remote SHA／3文書の内容照合後mandatory STOP。** 次担当はGPT／ユーザー：明示した算術条件を含む最小Q3-Nの実行要否と、後続の実caller接続を判断する。Codexは今回Q3-Nやcaller実装を開始しない。
