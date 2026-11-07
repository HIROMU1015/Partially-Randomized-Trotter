# 新A N3-B0-preflight：辞書パラメータ数値評価の準備

2026-11-08 JST、SERVER 2（GPUA100-gleap02）。基準 **`79dfa01e6796d1ecab50e30694f5b3f73b95f4ab`**、専用branch `codex/pr-reference-n3b0-preflight-20261108`。

**Stage A〜Dの静的preflightを完了し、mandatory STOP。** [PARAMETER_SELECTION_RULES](PARAMETER_SELECTION_RULES.md)と[NUMERIC_EXECUTION_SCOPE](NUMERIC_EXECUTION_SCOPE.md)を新規作成した。本turnではb／内部予算／基本key／coverageを数値化していない。実行承認の有無を証明する仕組みは作らず、ユーザーの最新指示にあるpreflight段階のSTOPを維持した。

## 1. 確認した固定条件と資料

Linear H4・1 Å・STO-3G・DF12、元H ground供給・状態準備費除外、Partial S2/LD3/h=.02/K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1/73 bits、最終native RZの一様strict epsilon=10^-11を変更しない。Qiskit1.3.0/compiler A、rz/sx/x/cx・全結合、B′4-CX controlled-RZZ、Reference F/G／actual SOURCE inverse／vacuum P／2 register／8-CX CSWAP、D融合／basis共有／scalar/event／X/Yを固定する。

energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720、199 shot/axisを維持する。W・相対位相・負角全連結wordのactual adjointを必須とし、phase fit、理想pi snap、無条件2pi同一視を行わない。

[N3-A REPORT](../pr_reference_n3a_static_20261108/REPORT.md)、[構成](../pr_reference_n3a_static_20261108/FINITE_DICTIONARY_CONSTRUCTION.md)、[誤差／停止](../pr_reference_n3a_static_20261108/STRICT_ERROR_AND_TERMINATION.md)、[T条件](../pr_reference_n3a_static_20261108/T_PRICE_FEASIBILITY.md)、[NEXT_SCOPE](../pr_reference_n3a_static_20261108/NEXT_SCOPE.md)を参照した。正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAPと既読の研究概要を適用する。

[N1](../pr_reference_n1_numeric_20261108/REPORT.md) `0a6ba582d246b3b724d42c8b56475c16ebef1dd9`、[CT-5B](../pr_reference_ct5b_20261107/REPORT.md) `0dbf1fe265176677e026ea982c0bd435987f4338`、[CT-4B](../pr_reference_ct4b_20261107/REPORT.md) `8012c3e9547690f0d722edac9d0b9a9391ba2f8e`と、[CT-0](../pr_reference_ct0_20261107/REPORT.md)、[CT-2](../pr_reference_ct2_20261107/REPORT.md)、[CT-3A](../pr_reference_ct3a_20261107/REPORT.md)の固定条件・未資格・STOPを再利用した。既存数理レビューやcompiler/context全監査を再実行していない。

## 2. 固定した選択規則

| 項目 | preflightで定義した規則 |
|---|---|
| L | CT-0保存m7のbinary64上界を厳密有理数として読む。上端到達や一様分布は仮定しない |
| b候補 | binary64で非ゼロの基本角を表現するための有限整数域0…1074 |
| 量子化項 | 既存192-bit算術のsqrt2区間上端をFractionへ移し、Q_b^up=c_hi*2^(-b-2) |
| 内部要求 | 各bについて、M_b*10^(-p_b) < epsilon-Q_b^upを満たす最小正整数p_b |
| 主仕様 | 成立候補のtuple(p_b,M_b,b)を辞書式最小にする。価格・cache結果を使わない |
| emitter | epsilon-Q_up-M*g_allocを**要求残余**として出力。実e_emitは未資格、0と仮定しない |
| 基本key | 選択後だけ、-b…J_bの2^jを生成・整数比照合。負側は全体adjoint規則 |
| 保存word | 固定3 archiveの既存recordだけ。実guard上界・word/hash/Wによる適合候補を列挙 |
| 既存native C0 | CT-4Bのe=-11採用wordを優先固定。内部候補で上書きしない |

これは数学が強制する一意の最適仕様ではなく、価格を見る前の決定的な研究仕様の選択である。precisionを不必要に厳しくしない優先順と、辞書サイズの二次目的を明示した。量子化・基本和を機械的に半分ずつ配分しない。数学的制約と採用規則を区別し、実T最適性は主張しない。

## 3. Emitterと価格の未資格

現[lower_native](../../../src/trottertracks/track_c2_b_verify.py)のWはQiskit global_phaseへ浮動加算される。exact wordのphaseと実emitterの一致を、新しい長い連結wordへ無条件で引き継げない。今回の数値scopeで具体化できるのは、候補仕様が要求する**e_emitのstrict上限**であり、実装がその上限を満たす証明ではない。

従って、後続にパラメータ値が得られても、実emitterの誤差資格はUNQUALIFIEDとして残す。残余が小さい場合に、新しいphase ledgerやemitterを自動実装しない。量子化と基本誤差は1 native RZの最終precision内部へ合算し、旧phase diskや合成diskを二重に計上しない。

N1のpointwise閾値173790、期待閾値`1602930468830332421754729/9223372036854775808`は保存値を引用する。実基本wordの価格がない間はB_dictが未資格であり、既存優先cacheを含む`max(B_dict,B_cache)<173790`の条件までしか言えない。今回も後続の限定パラメータ評価でも、T和／極値／期待価格を新たに計算しない。粗いcap不成立をReference一般の劣位としない。

## 4. 既存SOURCE・入力の再利用

| 資産 | 再利用方法と境界 |
|---|---|
| [track_c_h4_synthesis](../../../src/trottertracks/track_c_h4_synthesis.py) | 実binary target、exact_word_matrix／strict_guardのnorm規約、record形式を静的参照。acquire／guardは呼ばない |
| [track_c2_b_verify](../../../src/trottertracks/track_c2_b_verify.py) | written product逆順、W、実adjointを参照。emitterの新規実行はしない |
| [pr_reference_ct0](../../../src/trottertracks/pr_reference_ct0.py) | 保存recordの純粋metadata validator、旧phase域／precision条件。legacy method欠損を資格しない |
| [pr_reference_ct1](../../../src/trottertracks/pr_reference_ct1.py) | word pool／既存emitter照合・record provenance。stage runnerやword_matricesは起動しない |
| [pr_reference_ct4](../../../src/trottertracks/pr_reference_ct4.py) | C0採用表、保存strict archive、T/Tdaggerの意味、SOURCE。既存価格は変更しない |
| [pr_reference_rotation_v3](../../../src/trottertracks/pr_reference_rotation_v3.py) | literal angle／role、D／frame構造とsigned価格の区分を再利用 |
| [pr_reference_transfer_error](../../../src/trottertracks/pr_reference_transfer_error.py)／[pilot](../../../src/trottertracks/pr_reference_interference_pilot.py) | 保存LのIEEE演算順／remainder／F/G phase来歴。Lやvacuum生成を再計算しない |
| [区間helper](../../../src/trottertracks/track_c_compiler_a_18round.py)／[有限bit helper](../../../src/trottertracks/track_c_h4_finite_bits.py) | Iと区間端点の整数比化という既存規約。確率生成やsamplerは使わない |
| [N1 SOURCE](../../../scripts/run_pr_reference_n1_numeric.py) | pinned git読取、Fraction会計、metadata関数を副作用なしに使う方式を参照。N1 runnerは再開しない |

保存入力と必要SOURCEのcommit/path/SHA256は[INPUT_REFERENCES.json](../../../artifacts/pr_reference_n3b0_preflight/2026-11-08/INPUT_REFERENCES.json)。CT-4B／旧C2-B／CT-1のBUNDLESとarchiveの保存先・schemaを読取確認した。既存wordを新guardで修復しない。科学moduleをimportしていない。

system pythonにmpmathがないことをdistribution metadataで確認したため、[CT-4B保存ENVIRONMENT](../../../artifacts/pr_reference_ct4/2026-11-07/ENVIRONMENT.json)の既存project専用venvを数値runtimeとして特定した。現存する同pathのPython3.12.3／mpmath1.3.0もmetadataで確認済み。sqrtやguardは呼ばず、venv／共有package／設定は変更していない。

既存にN3-B0のb選択とcoverageをまとめたrunnerは確認できない。後続で必要なら、標準整数／Fractionと既存metadata規約を使う小さい**パラメータ会計SOURCE**をfresh領域に追加する理由がある。これはN3-B1の丸め／連結／emitter policy実装とは異なり、量子回路・wordを生成しない。

## 5. 資格・完了・次担当

preflightの仕様・入力対応・必要計算・検算・STOPを具体化した。数値のPARAMETER_SCOPE_FIXED、coverage件数、DICTIONARY_ACQUISITION_REQUIREDの対象数はまだ出していない。実辞書・全support word policy／取得停止・価格上界・CT RPE成功・18round総T／T-depthはUNQUALIFIEDのまま。

新規科学数値評価、b／内部予算／basic key生成、coverage計算、合成、guard、word連結、build/compile、sampling/simulation、T価格／期待／極値、最適化はいずれも0。旧SOURCE/raw/result/grant/ledger/marker/STOP、共有設定・他ユーザー・別Trackは変更していない。64 CPU利用可能なSERVER 2の静的読取／文書作成だけを行い、独自資源capは追加していない。

**preflight完了でmandatory STOP。** ユーザーの数値実行承認は受領済みであり、再度の承認認証やgrant発行を求めない。本書で作ったscopeはb／内部誤差要求／基本key／保存coverageという範囲で整合するが、実emitter資格・N3-B1実装・合成・T価格取得へは広げない。

次担当は**GPT／ユーザー**：今回明文化した選択優先順と条件付きemitter残余を確認し、数値段階の再開を判断する。**Codex**：その指示がある場合に固定scopeの限定数値処理を実行・公開し、完了時STOPする。未固定条件を推測した数値実行やN3-B1への自動移行は行わない。
