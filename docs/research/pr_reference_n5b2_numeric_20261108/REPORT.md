# 新A N5-B2：保存r32中央角の限定logical成分評価

2026-11-08 JST、SERVER 2。Scope基準 **50013be49f03cb609fc6e43ed4dfcf43251f7823**。Branch `codex/pr-reference-n5b2-numeric-20261108`。

**固定4 positive literalの全8 axis比較で、現b36のlogical成分上界Q+Gは保存capを超えた。全4角でQがGより大きい。指定検算を完了しmandatory STOP。**

これは**現b36・保存guardの当該十分証明では足りない**という有限sliceの負結果であり、実word誤差の下界、m17のRPE失敗、新規合成・特定の新bが必須という認定ではない。新しいprecision／domain／policyや候補Gの正式な誤差規約を採択していない。

## 固定範囲と条件

[N5-B2 NEXT_EXECUTION_SCOPE](../pr_reference_n5b2_static_20261108/NEXT_EXECUTION_SCOPE.md) §2–3と[NEXT_NUMERIC_INPUTS](../../../artifacts/pr_reference_n5b2_static/2026-11-08/NEXT_NUMERIC_INPUTS.json)の中央sliceだけを実行した。同JSON内のbase_commit f250bdb…は静的scope作成時の科学SOURCEを指す。今回のscope commit50013be…への追加は静的成果物で、既存policy／43 record／m17科学入力は変更していない。

背景条件はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、既存D融合・basis共有・scalar/event・X/Yを維持する。

M17 q131072／r32／88 bits、B′626／Reference630 shots/axis、alpha1/720／failure.05／phase cap.4／energy1.5936e-4 Haも変更しない。今回これらの物理回路・state・RPEを評価したわけではない。

Policyは**既存のまま**b36、j=-36…6、各保存basic wordのactual guard、最近接ties-to-even、ascending written orderを参照した。C0優先を変更せず、別precisionや別targetへ拡張していない。対象は保存r32中央の4 positiveと対応negativeだけで、新しい到達集合ではない。

## 計算方法

実binary64 thetaに対し、固定round_indexが返すnと選択basic集合Jを用いて

`delta = |theta - n / 2^36|`,

`Q = saved_sqrt2_upper * delta / 2`,

`G = sum_(j in J) saved_guard_j`,

`e_log_component_upper = Q + G`

を評価した。保存sqrt2上端と各basicのoutward guardをそのまま使用し、新しいstrict guardは実行していない。この上界はexact Wを含むlogical productへのもの。Emitter／consumerの具象化誤差を計算したものではなく、既存emitted certificateの値は変更しない。

全成分とcap判定はPythonの正確なFraction／整数で行った。保存capはJSONのbinary64値を正確な有理数へ変換し、hexも保存した。結果JSONの24桁小数表示は上方向丸めで、判定の正本はexact_fractionと保存capの有理数である。

## 結果

以下の成分表示は概数。全桁・正確な分数・selected j・index・負角対応・8比較の正本は[RESULTS.json](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/RESULTS.json)。

| Arm | Positive literal | Q（約） | G（約） | Q+G（約） | X/Y cap照合 |
|---|---|---:|---:|---:|---|
| B′ | 0x1.7e4088e024c5fp-12 | 1.280446e-12 | 1.578007e-13 | 1.438247e-12 | 両軸ABOVE_CAP |
| B′ | 0x1.fdab627b8d37ap-14 | 3.903359e-12 | 2.097838e-13 | 4.113143e-12 | 両軸ABOVE_CAP |
| Reference | 0x1.7e4088e024c5fp-11 | 2.560892e-12 | 1.440817e-13 | 2.704974e-12 | 両軸ABOVE_CAP |
| Reference | 0x1.fdab627b8d37ap-13 | 2.483040e-12 | 2.026609e-13 | 2.685701e-12 | 両軸ABOVE_CAP |

| Arm | 保存cap X | 保存cap Y |
|---|---:|---:|
| B′ | 6.902168834414281e-16 | 6.902168833913054e-16 |
| Reference | 2.5648979455928107e-15 | 2.5648979454057876e-15 |

Capは旧worst-case一様会計で保存Nを保つ**十分割当**であり、最適precisionや必要条件ではない。全8比較はABOVE_CAP、EQUAL_CAPとSTRICTLY_BELOW_CAPは0件。保存論理誤差成分のうちQ>Gは4/4であり、現在の証明ではemitter scalarだけが唯一の未解決項ではない。

ただしQとGは保守的な上界で、真のword誤差がQ+G以上であるという意味はない。別の誤差配分やより鋭い保証を排除していない。今回、Q／G単独を使った追加のcap比較、guard再評価、新b探索、全位置の誤差和は行っていない。

## 検算とSOURCE再利用

- 保存43 recordをmanifest／archive member／record hash／sequence hash／strict spec／saved guard／up_to_phase=Falseへ照合。Word文字列は読み取りとhash確認だけで、連結・生成していない。
- 4 positiveと4 negativeの**合計8回**だけ、変更しないround_indexを実行。負角はn_negative=-n_positiveで、binary64 targetの符号対称性も一致した。
- 各positiveの選択bitからnを再構成し一致。既存域・最大index・bit長内、最近接half-cellとties-to-even条件、Q+Gの成分恒等式も全件PASS。
- 4 positive×X/Yの8 cap比較を正確な有理数で完了。Negativeのlogical boundはcomplete adjointと符号対称性から同じで、新しいword／行列を作って確かめる処理はしていない。

[限定wrapper](../../../scripts/evaluate_pr_reference_n5b2_central.py)は、既存[dyadic provider](../../../src/trottertracks/pr_reference_dyadic_dictionary.py)のround_index bodyと、provide内の**indices／selected／delta／q_upper／g_upperの5代入を変更せずASTで再利用**した。Provider全体はword連結・T/W集計を含むので呼ばず、他の科学moduleもimportしていない。追加部分は保存guardだけの入力binding、成分和、検算・比較・出力であり、新しいprovider実装ではない。

[INPUTS](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/INPUTS.json)、[SOURCE_INPUTS](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/SOURCE_INPUTS.json)、[ENVIRONMENT](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/ENVIRONMENT.json)、[CHECKS](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/CHECKS.json)を保存。再現入口は `PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_pr_reference_n5b2_central.py`、固定Git入力とfresh出力領域が条件である。STOP後の再実行を指示するものではない。

実処理は約0.065秒。4つの小さい有理式で、bit選択／成分の逐次依存と最初の入力不一致でSTOPする範囲のため単一processで完了し、独自CPU/RAM capや共有設定変更はない。正式共通指示2026-10-25、親／worktree AGENTS、PROJECT_MAPと既存研究資料を適用。著者Codexの限定数値診断であり独立科学レビューではない。

## 範囲外と次の判断

新規合成、strict guard、word生成、provider／emitter／consumer実行、native build/compile、Operator、simulation／sampling、RPE再会計、T価格評価は0。旧SOURCE／raw／result／grant／ledger／marker／STOPと他Track・統括兼Dの作業は変更していない。

M17全supportのG適用、logical policy、vacuum／scalar／event域、保存626／630の十分性、T優越は引き続き未資格。今回の数値結果から新しいb・基本精度・domain方式を自動選択していない。

**[MANDATORY STOP](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/STOP.json)。次担当はGPT／ユーザー**：この有限な負結果を踏まえ、precision／域／policyを再設計するか、保留するか判断する。追加の科学実行が必要なら、その固定scopeと明示承認を別途定める。Codexの後続探索はない。
