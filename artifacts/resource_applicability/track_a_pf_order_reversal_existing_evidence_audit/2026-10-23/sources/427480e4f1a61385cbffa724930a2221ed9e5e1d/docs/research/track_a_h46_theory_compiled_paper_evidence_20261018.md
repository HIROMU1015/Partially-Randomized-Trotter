# H4/H6理論・compiledリソース比較：論文化向け証拠整理

状態：**EVIDENCE_DOSSIER / 中心claim未採択 / 完了後mandatory STOP**。

この資料は、固定7候補の既存SOURCEと修正済み結果が、どの記述をどの範囲で支持するかを整理する。
論文原稿、新規性の判断、正式な主図、一般的なPR優位性の採択ではない。
期待費用対有限標本平均、同一event、shot込みpair比較を、異なる評価対象として扱う。

[表・全行索引・Claim–Evidence Matrix・来歴・検査の入口](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/README.md)。

## 1. 研究の目的と対象範囲

DF-native部分ランダム化積公式（PR）と決定論的積公式（PF）について、宣言した理論ゲート分解モデルと、
指定compilerによる保存compiledゲート数を比較した研究である。
対象は線形H4/H6、隣接間隔1.00 Å、STO-3G、T=0.8の固定7候補。
資源指標は状態準備を除くfull Hadamard wrapperのRZ/CX countと、保存shotを掛けた総RZ/CX費用。
古典計算のwall・RSS、logical qubit、T-count/T-depth、surface-code資源を改善した研究とは扱わない。

数値の正本は結果commit **`6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1`** の
[修正版Stage 3](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/result_summary.json)。
旧表は変更履歴と旧図の説明にだけ用いる。α/β、H8、Track B、他geometry、別H6確認標本は混合しない。

## 2. 理論モデルと費用指標

[固定counter SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f01fa5fbd9c508e468dac3cff8405f57370dbb51/src/trotterlib/native_df_gate_model.py)
の`assemble`、`partial_prefix_circuit`、`expected_rte_occurrence`と、
[adapterの`native_expectation`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f01fa5fbd9c508e468dac3cff8405f57370dbb51/src/trotterlib/pr_theory_compiled.py)を基準とする。

|モデル|保存SOURCEの規約|解釈の限界|
|---|---|---|
|A|各Gaussian sandwichを出力し、隣接basis境界の任意の再利用を無効にする。source由来のphase集約とidentity barrierは共有する。|「すべての最適化がない回路」や実compilerの上界ではない。|
|B|厳密に等しい隣接basisに対して、定義済みの境界相殺・再利用を行う仮想モデル。明示barrierで閉じる。|近似的なbasis一致、可換変形、角度融合、任意のtail/prefix境界相殺を追加しない。|
|NATIVE_MIXED|PRのprefix A＋各closed finite-RTE tail B。native scalar、control、wrapperに合わせる。A/Bの総数を補間したものではない。|全回路をBで最適化したものではない。PFはtailを持たずAと同じ。|

宣言したlowering profileは`explicit_rz_sx_x_cx_v1`、S4 schedule互換性profileは`native_binary64_s4_v1`。
これらのcountと、Qiskit lowering＋optimization level1後のcountを分ける。
理論countをcompiled countの厳密上界とはしない。

保存表の算術定義は、既存SOURCE
[`scripts/h46_cost_summary.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8eb64ba8e88c219a6b707e1c72d130f5f37fcb/scripts/h46_cost_summary.py)
のまま記載する。

- `rho = compiled / theory`。
- `residual = (compiled − theory) / compiled`。通常の「theoryを分母にした相対誤差」と取り違えない。
- `G = Σ_axis N_axis C_axis`。PR/PFそれぞれの保存shotを用い、同じshot数へ置き換えない。
- 区間比は保存されたlower/upperを使用する。中点化せず、信頼区間とも呼ばない。

本作業ではこれらの式を数値評価せず、既存欄を抽出・転記した。

## 3. 固定された物理系・回路・compiler・標本条件

|系|DFとmode|固定候補|保存PR標本|
|---|---|---|---|
|H4|legacy DF rank12、8 system modes|PF `B1_p12_q1_2nd` / `B1_p12_q1_4th`、PR `B2_p3_q1_2nd_r4_K2` / `B2_p3_q8_2nd_r2_K4`|anchor n32|
|H6|tol-only DF rank19、12 system modes|PF `B1_p19_q8_4th`、PR `B2_p10_q8_2nd_r1_K2` / `B2_p15_q8_2nd_r2_K4`|探索n8|

全候補はlinear chain、1.00 Å、STO-3G、T=.8。
決定論的PFはn1。same-event表は各候補1つの保存event／固定回路であり、PRのn32/n8平均ではない。
H6の別研究の独立確認n32は今回の固定panelへ混合していない。
DF生成政策が異なるため、H4とH6を単純なサイズ依存性の検証と呼ばない。

controlはsymmetric-directional、full Hadamard wrapper、cosine/sine、状態準備費用P=0。
Qiskit1.3.0、basis `rz,sx,x,cx`、optimization level1、transpiler seed17、backend/coupling map/routing topologyなし。
compiler settings hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

[Table Aの機械可読条件](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/table_a_conditions.json)
に各候補のH/state/DF/input hash、元標本phaseとSOURCEを対応付けた。

## 4. 検証の経緯と最新SOURCE

|段階|結果commit|実施済み証拠と歴史上の位置|
|---|---|---|
|元比較|`7e454b91185ed7b0f92a3790f800ba5d65d5f156`|元科学SOURCE `536a3671c0ec2939ff3886e40467db26a27ec65a`。84/84/360行、S4の8行は当時UNATTRIBUTED。|
|静的・人工原因調査|`b7385211e789ce9e9ab9515f3d5779fbd117bfb6`|1 ULP→厳密zero選別の機構。16人工tests。実block帰属はこの時点では未確定。|
|実block別検証|`cb501906269ef31fe8733a3391db946b34467163`|SOURCE `4d60519e11ecf70a2d19f536ce69ed95b314724c`、保存66 primitive評価で8行の局所原因を照合。|
|修正・人工回帰|`a139250850bd9eee313162454affbbd46e2fa6fa`|SOURCE `f33d917f10c5f92f63b23776387cbd16e85abfd1`、68人工tests。|
|修正counter受入|`fcb937a0c56ee75e2d5d32a9472357f3f346aaa8`|実counter評価SOURCE `f01fa5fbd9c508e468dac3cff8405f57370dbb51`、8整数行・33負stage primitive receipt一致。|
|修正版比較正本|`6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1`|再会計SOURCE `da8eb64ba8e88c219a6b707e1c72d130f5f37fcb`、84/84/360行。|

数学的に等価な負係数`-a*(1/(2-a))`と`-a/(2-a)`、stage時間の演算順がbinary64で異なり、
元producerの厳密zero判定に作用した。実block証拠ではH4 fragment1の保持RZが正味+1、
H6 fragment5が正味−1で、q反復とdirectional展開によりRZ/CX差+2／−16へ対応した。
これは数学的PFの誤りではなく、counterのnative binary64互換性の問題。
元native生成器、producerのzero規則、compilerを変更した修正ではない。

互換性修正はQiskit前の8行count差を解消した。
そのことと、理論lowering対Qiskit lowering＋opt1の残差の説明を分ける。
旧報告書や表のUNATTRIBUTEDを上書きせず、新しいcount受入証拠で解消範囲を説明する。
最新表にも継承されたUNATTRIBUTED欄があり、全てを「S4原因未確定」と読んだり、全残差を「解決済み」と読んだりしない。

元Stage2/3の実行SOURCEはそれぞれ`98e5c9e61f245dbb9faf381d835a3846ee7e9bf3`／
`34b2da1a6bfc001314e0e31458d803caca80adf2`、公開同一tree mirrorは
`bdbd8d5b4ca00d2d0b49c6829248419c86b4f26d`／`fe889c4ac070a95ba949370ebf523a0749cc5149`。
mirrorを実行SHAとは呼ばない。S2/PRの再利用行には元の評価SOURCEが残り、
全行を修正counter SOURCEで新評価したことにはしない。

## 5. 修正済みの主要結果

### 5.1 三つの比較単位

|正本|行数|評価対象|含まれる差と解釈|
|---|---:|---|---|
|`expectation_vs_saved`|84|理論期待費用lower/upper 対 保存compiled有限標本平均|PRではモデル／compilerの差に加え有限標本のevent構成差を含む。PFは固定回路。CI・完全な数値envelopeではない。|
|`same_event_comparisons`|84|同じ保存eventの理論／native explicit／compiled|各候補1 event。event構成を揃えたcount比較。n32/n8平均の再検証ではない。|
|`pf_pr_comparisons`|360|保存shot込み理論期待費用比 対 保存標本compiled費用比|5精度・2 allowance倍率・3モデル・2metric、固定6pair。不適格は欠測として保持。|

S4修正で変わった行は24／24／168。
保存70 shot条件、bias・normalization・eligibility、S2/PR、compiled値は不変。
共通適格192行中の方向不一致6行は、同一H4/RZ条件の3モデル×2倍率という相関した記録。
6つの独立実験ではない。旧→新の方向変化は保存監査で0件。
これらは保存された集計値の転記で、今回の新規統計集計ではない。

### 5.2 H4の方向不一致候補

PR `B2_p3_q8_2nd_r2_K4`、PF `B1_p12_q1_4th`、ε=.05、倍率1、P=0、RZ。

|値の種類|保存報告のPR/PF表示値|
|---|---:|
|修正前理論NATIVE_MIXED（履歴）|0.8716682443|
|修正後理論NATIVE_MIXED（正本）|0.8716321313|
|保存compiled標本費用|1.0393357766|

表示値は最新固定報告書から転記した。有理数の正確な値は`COST-233`と正本JSON `/233`。
同じ保存shotのもとで理論はPRを低コスト、compiledはPFを低コストとする。
S4修正後もこの方向不一致が残る。これは固定条件付きの主張候補C02/C04であり、
PR/PFの一般的順位、統計的有意性、全候補の最適性を証明しない。
CXには別の保存値があり、このRZの結果をCXへ移さない。

### 5.3 count整合と残る差

修正後counter対保存nativeの受入は8/8整数行で一致し、
same-eventのNATIVE_MIXED対native explicit countは28/28 metric行で一致した。
一方、同一eventのcompiled/NATIVE_MIXED比は、最新報告で約0.448721〜0.638070。
この範囲も保存報告から転記し、新しいmin/max計算は行っていない。

この固定panelの観測を普遍的なcompiler変換係数や厳密boundとしない。
Qiskitのloweringとopt1の寄与は個別には分離されていない。
期待費用表とsame-event表で比率の評価対象が異なるため、両者の幅を混ぜない。

## 6. 論文に使用可能な表の候補

[表候補・比較単位・全元rowの索引](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/table_index.md)。

|候補|内容|元データと抽出方針|配置案と理由（未採択）|
|---|---|---|---|
|A|物理条件・7候補・モデル・compiler・標本|条件と元candidate receiptを転記|本文の条件表又は補足。再現範囲を最初に明示する。|
|B1|期待費用 対 保存平均|全84行を索引化。表示候補はmixed28行、A/Bも正本へリンク|補足の数値表。母平均保証と混同させない。|
|B2|同一eventの理論/native/compiled|全84行を索引化。表示候補はmixed28行|本文又は補足。event構成を固定した差を示す。B1と別表・別caption。|
|C|shot込み固定pair比較|全360行を索引化。mixed/RZ/倍率1の全30条件を抽出し、不適格も残す|本文では指定例、全pairは補足。全方式のwinner表とは呼ばない。|
|D|S4修正前後|8受入行と対応する新旧same-event値。A/Bにも及ぶ変更は正本差分表を参照|技術付録。count互換性修正と科学的費用比較を分離。|

数値表の全欄は正本JSON値のdeep copy又はpointerで、ratio・残差・shot・方向の再計算はない。
小数表示用の新しい丸め数値や、期待値区間の代表中点を生成しない。
今回のrow IDは文書索引用のaliasであり、元科学データに新しいevent IDを付したものではない。
528行全てに固定commit、JSON pointer、row digest、実評価SOURCEを付ける。

## 7. Claim–Evidence Matrix

中心成果物は [人間向け一覧](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/claim_evidence_matrix.md) と
[全欄JSON](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/claim_evidence_matrix.json)。
12候補を日本語・中立的英語案、条件、SOURCE、RESULT、元row、証拠種別、支持範囲、支持しない範囲、
未解決事項、追加証拠要否、正式採択要否へ対応付けた。

C01/C08は同一eventの局所観測、C02は保存shot・有限標本を条件とする方向不一致、
C03/C10は技術的count整合、C04は修正前後の保存方向監査、C05は定義済みモデル差。
C06/C07の一般的優位性・母平均保証には証拠が不足する。
C09/C11/C12のopt1単独帰属・厳密上界・6独立実験という断定は支持しない。
どの候補も正式な中心claimとして採択していない。

## 8. 図候補と元データの所在

[既存2 SVGと将来4設計案](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/figure_inventory.md)。
旧SVGは元結果commitの`expectation_vs_saved.json`、`same_event_comparisons.json`を描いている。
S4修正前の図であり、最新数値を反映した図ではない。
本固定研究に正式採用された主図・完成原稿が存在すると仮定しない。他研究の原稿の有無も判断しない。

設計案は同一event count、期待費用対平均、shot込みpair比、S4新旧差の4種。
横軸・縦軸・系列・比較単位・元ファイル・誤解しやすい表現を一覧化した。
既存値を表示するだけなら追加の科学計算は不要だが、今回は描画・正式採用を行わない。

## 9. 未解決事項と主張への影響

[13事項の影響一覧](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/limitations.md)。

固定研究を解釈する上で直接必要な制約は、DF政策差、PR有限標本、経験的allowanceと保存shot、
P=0、理論区間の意味、lowering/opt1の未分離である。
これらは保存条件内の値を否定するものではなく、何を比較した結果かを限定する。

より強い一般化・再現・同値性の主張には、母平均／希少event上界、panel外の最適性、
旧compiler binary完全再現、全wrapper word bit一致、状態準備、hardware/noise/routing、
耐故障論理資源の追加証拠が必要となる。
H6確認n32を混ぜれば自動的にこれらの制約が解消するわけではない。

追加証拠案は目的と対象だけを示し、計画の正式採択、件数予約、grant発行、新計算は行っていない。
S4の8行不一致は既に解消しており、「未解決事項」に戻して再実行しない。

## 10. 支持できる主張／支持できない主張

使用候補となる事実は、固定回路・compiler下の構成依存差、特定H4条件での方向不一致、
S4有限精度互換性の原因とcount整合、修正後も残るcompiled残差である。
期待費用・有限標本平均・同一eventのどれについての記述かをcaptionにも明示する。

一般的PR優位、候補間の最適性、母平均保証、理論値の厳密compiled上界、opt1単独の効果量、
無条件の精度一致・end-to-end量子資源改善は今回の証拠から採択しない。

## 11. ユーザーによる科学的判断が必要な事項

[判断事項5件](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/user_decisions.json)。

1. 独立論文にするか、PR研究の一節／補足技術検証とするか。
2. 構成依存差・方向不一致・技術整合のどれを中心に据えるか。
3. A/B/C/Dと図設計案を本文・補足・技術付録へどう配置するか。
4. 現在の固定範囲に限定するか、強い主張のための追加証拠を別途求めるか。
5. 次に原稿草案又は候補図を作成するか。

上記は選択肢であり、論文のストーリー、新規性、投稿先を決定していない。

## 12. 次段階の承認と終了

今回の技術検査は固定ファイルbytes/SHA・SOURCE closure・文書schema・link・528行参照・抽出値一致を対象とする。
結果は [検査記録](../../artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/technical_checks.json)。
科学codeをimport・実行せず、科学arrayを読み込まず、理論費用・比率・shot・順位を再評価しない。
新しい科学計算、再解析、compile、sampling、科学grantは0件。
primary agentの技術自己検査であり、独立別担当review・GPT/Claude詳細科学レビューではない。

既存SOURCE・compiled結果・grant・ledger・marker・STOPは変更しない。
新規資料は独立branch `track-a-h46-theory-compiled-paper-evidence-20261018`へ固定し、
許可済みrepositoryへ公開・remote SHAと取得bytesを確認する。文書固定commitは最終チャットで報告する。

次の担当は**ユーザー**。上記の主張・配置・追加証拠の要否を判断する。
**Codex**による原稿執筆、新図生成、追加科学計算は別途指示・必要な実行承認後の作業である。
公開確認後に**mandatory STOP**し、自動的に次へ進まない。
