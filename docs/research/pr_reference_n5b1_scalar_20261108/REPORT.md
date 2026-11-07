# 新A N5-B1：共通joint scalarのSOURCE対応・限定runtime検証

2026-11-08 JST、SERVER 2。固定基準 **624646b9a4782a65e90561ea29df6beccea2a42c**。Branch `codex/pr-reference-n5b1-scalar-20261108`。

**固定SOURCEは、word具象化時の浮動global phase差をwhole jointの共通scalarへ分離する条件を支持する。固定10 word/composite・160組の局所fixtureもruntime適合を確認した。承認範囲を完了しmandatory STOP。**

これはlogical wordとRZ targetの近似差を消す結果ではない。新しいstrict emitter上界、precision、provider、RPE誤差会計を採択していない。**m17のRPE成功・保存shot十分性・T優越は未資格**。

## 判定

| 判定 | 到達点 |
|---|---|
| JOINT_SCALAR_FACTORIZATION_SOURCE_SUPPORTED | 固定lower_native／consumerとnamed ideal gateモデルの入力クラスについて、位相欄と同じnamed productの分離をSOURCEから確認 |
| LOCAL_RUNTIME_COMPATIBLE | 固定160組でnamed列・wire・actual adjoint・reset/readout／clbitが整合。予測scalarとX/Y局所診断が旧tolerance内 |
| CONDITIONAL_ON_FULL_CONSUMER_PATH | 本番の全入力・provider・sinkまで同じ前提を保つことが必要。m17 wholeはbuildしていない |
| SIGNAL_FACTOR_QUALIFICATION_INCOMPLETE | m17全supportへ適用する研究上の保証・policy・域を今回正式採択していない |
| RELATIVE_PHASE_OR_GATE_MISMATCH | 採用実行では該当する不一致なし |
| M17_RPE_UNQUALIFIED | Raw disk・半径・N・Tを再会計していない |

## 固定条件と実行範囲

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler Aを維持。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event位相・X/Yは不変更。

M17 q131072／r32／88 bits、保存NはB′626／Reference630 shots/axis、alpha1/720／failure.05／phase cap.4／energy1.5936e-4 Haは背景条件の引用である。今回これらの物理回路・ground・trajectoryを生成していない。

[固定JSON](../../../artifacts/pr_reference_n5b0_static/2026-11-08/NEXT_DIAGNOSTIC_INPUTS.json)のC0 W residue0〜7各1 wordと、frame `0x1.08ba32d420f2cp+2`／b2 `0x1.921fb54442d18p+2`だけを使った。Positive wordは保存文字列、負側はcomplete actual adjoint。b2のW=72とresidue0を区別し、角度mod／snap／phase fit／新controlを導入していない。

各wordで正負、zero／one_tenth／prev_tau／tau_via_negative_subnormal、CX前後の16組、合計160組を検査した。Fixtureは2q/3c、prefix phase、reset(1)、SX(0)、SX(1)、wordとCX(0,1)の前後、SX(1)、measure(1,2)／measure(0,0)。入力|00>でprefix reset後のbodyを評価し、`X(q1) tensor I(q0)`／`Y(q1) tensor I(q0)`（little endianの行列順）を診断した。実RPEのH4入力を代用したものではない。

## 技術結果

結果の正本は[LOCAL_RUNTIME_CHECKS.json](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/LOCAL_RUNTIME_CHECKS.json)。新しい数値はこの小fixtureの診断だけである。

| 項目 | 結果 |
|---|---|
| 完了fixture | 160/160、全件PASS |
| Named gate／wire／readout・clbit | 既存expected_stream／check_consumerと全件一致 |
| 正負 | 保存positive word／complete inverseに一致。Signed zeroは元SOURCEどおり |
| Scalar | Exact Wとnative／outerの実phaseから導出。行列に合わせたfitなし |
| 予測scalar込み行列残差の最大観測値 | 3.436765120009076e-13 |
| X/Y信号差の最大観測値 | 4.674038933671909e-13 |
| 旧local診断tolerance | 1e-10 |
| 同じnamed streamのOperator評価 | Cacheで37通り。Logical word行列もcache |
| 採用実処理時間 | 約12.10秒 |

上の値はbinary64行列積等を含む**技術診断の観測値**で、実emitterのstrict誤差上界・下界でも、10^-16級certificateでもない。Wを除いた同じnamed productと独立なexact-word参照を比較し、scalarをSOURCE式で指定することに意味がある。有限数値の一致だけから一般信号不変性を認定していない。

## SOURCE上の意味と限界

[JOINT_SCALAR_FACTORIZATION](JOINT_SCALAR_FACTORIZATION.md)は、すべてのcontrolを構成した後のjoint native回路を開始点とする。Logical body `L=exp(i*phi_L)A` と実emitter body `E=exp(i*phi_E)A` は、同じnamed product Aを持つ。Scalarのwire embedding、CX前後、complete inverse、inplace joinを通じて `E=exp(i*chi)L` と書ける。同じreset／readoutの測定ではscalarが消え、trajectoryごとのX/Y恒等式から同じlawの平均測定信号へ接続する。

**Strict operatorでは一般にEとLは異なる。** 現certificateの64u・辞書emitter・shellを0へ書き換えていない。SOURCE scalar／event／vacuumのrelative角、logical guard、量子化、旧raw diskを除外していない。[SIGNAL_INVARIANCE_ASSESSMENT](SIGNAL_INVARIANCE_ASSESSMENT.md)に帰属とm17の不足を分離した。

## 再利用・再現・保全

[SOURCE_PATH_VERIFICATION](SOURCE_PATH_VERIFICATION.md)と[SOURCE_INPUTS](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/SOURCE_INPUTS.json)に、固定関数・record・hash・再利用箇所を記録した。N3/N4の同じ目的の単独検査と保存whole projectionは再実行せず、今回のjoint templateに不足する部分だけを実行。新wrapperの保存literal bindingはword／certificateを生成しない。

[INPUTS](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/INPUTS.json)、[ENVIRONMENT](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/ENVIRONMENT.json)、[CHECKS](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/CHECKS.json)を保存。再現入口は `PYTHONDONTWRITEBYTECODE=1 <ENVIRONMENT記載のvenv>/bin/python scripts/verify_pr_reference_n5b1_scalar.py`。固定commit・fresh出力領域が条件であり、STOP後の再実行を指示していない。

初回のnamespace不足による技術エラー（json未渡し）はSOURCE／出力／STOPごと[保全archive](../../../artifacts/pr_reference_n5b1_scalar/INITIAL_WRAPPER_ERROR_20261108.tar.xz)に残した。Initial partial consumer実行1件と、修正後の採用160件を区別する。実装不一致を隠した再試行ではない。

新規合成・strict guard・precision変更・production provider実行・native PF build/compile・H4 simulation/sampling・m17 RPE再会計・T価格評価は0。著者Codexの技術検証であり独立科学レビューではない。正式共通指示2026-10-25、親／worktree AGENTS／PROJECT_MAPと既存研究資料を適用。共有設定・他ユーザー・統括兼Dの作業・旧SOURCE／raw／result／grant／ledger／marker／STOPは不変更。

**[MANDATORY STOP](../../../artifacts/pr_reference_n5b1_scalar/2026-11-08/STOP.json)。次担当はGPT／ユーザー**：[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)の保証経路とlogical policy要件を判断する。候補Sの数値改善、新precision／辞書、m17再会計、価格評価へ自動進行しない。
