# PR資源評価スライドQ&A：Codex参照メモ

最終更新：2026-10-22 JST。Q&Aは同じファイル内で78件から96件へ更新した。Q01–Q78のanchorは維持し、追加はQ79–Q96。ユーザーの「追加しておいて」と継続するcommit・push指示に基づく文書更新である。

## 今回の編集範囲と現行資料

- 読み手向けの基準：[発表用14枚・圧縮12枚・補足8枚](PR資源評価_発表用スライド構成案_14枚基本・12枚圧縮_補足8枚_2026-10-22.md)。編集前の公開基点は `9b042caa3eaebcf023b3e74e8327990368a6e6e4`。
- 詳細説明の保存先：[説明意図明確化版](PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_説明意図明確化版.md)。発表用MD・詳細版・全図は今回変更しない。
- 今回更新するのは[Q&A](PR資源評価_スライドQAノート_2026-10-21.md)、本メモ、[案内README](README.md)のみ。通常編集は同じpathを使い、改訂ごとのQ&Aファイルは作らない。
- 公開先は `codex/pr-resource-slides-supplement8-20261022`。継続する公開承認は[発表用メモの運用記録](2026-10-22_PR資源評価_発表用補足8枚_Codex参照メモ.md#slide-publication-policy)を参照。新規科学計算・科学的受入・図の本編採用を伴う承認とは区別する。

## 追加質問と既存回答の同期

|質問・変更箇所|確認内容・編集判断|
|---|---|
|Q79–Q81|PFも具体的な回路実装を反映した解析会計であり、原論文のPFだけが全回路実compileという根拠はない。Appendix E.3の式を転記し、G_detをstage全体の単価としない|
|Q82–Q84、既存Q58|ユーザーの目的は、回路を構築しない数式評価とcompiled費用の差。原論文の数値再現・Tailのみの分離は必須ではなく、有限MCの差を含むことは認める。全差をTailやcompilerだけへ帰属しない|
|Q85–Q88、既存Q27/Q59|同じ保存Nでの比較はC置換によるG差。現在のNは数値bias・有限RTE正規化・経験的uとHoeffdingの混合会計。理想同値ならcompileだけでNは変わらない。原論文のRPE round別N_mと固定Tの軸別N_aを同じ契約の式と呼ばない|
|Q89–Q90|新しいCNG資料の20条件を参照。N_circuit・N比・G10/G11・N寄与は未取得nullであり、N差ゼロという結果ではない。H6/H8は条件付き参考値。本編6への採用は未判断・未実施|
|Q91–Q92、既存Q07/Q53–Q54|outer32は各outer stepのRTE小ステップ合計32。S2ではr32相当、高次では各区間への割当の合計32。T/qから出る値やMC本数ではない|
|Q93、既存Q14–Q16・対応表|「主比較8・9」は14枚版のスライド8・9、圧縮版7・8。探索A各n8とouter32・独立確認を区別|
|Q94、既存Q20/Q69|本編13は日本語の探索→確認表へ変更済み。H4粗PR→PF、H6粗PF→PF、H8はPOINT_REPRODUCED。S5の固定候補・本数と元Dも保全|
|既存Q39|本編1の既存Ha換算表示5行を転記。単一固有状態・半径1・既知branch・ℏ=1、T=.8、7桁丸め。新しい換算計算なし。達成エネルギー精度や丸め込みの厳密上界とはしない|
|Q95–Q96、既存Q74/Q77|発表用S1–S8と詳細A.1–A.9/B.1–B.23を区別。発表用別MDは原本保全・別版作成という後続指示による。公開状態と過去の未公開履歴を区別|

## 今回照合した固定資料

### 原論文と旧理論／compiled比較

- 原論文：[Phase estimation with partially randomized time evolution, arXiv:2503.05647v1](https://arxiv.org/pdf/2503.05647v1)。ローカル原本 `Phase estimation with partially randomized time evolution.pdf` の同版本文も参照。Appendix E.1（PDF pp.36–38）でPFのPauli回転・CNOT相殺・lexicographic ordering・DF基底変換等、E.3（pp.39–40）でG_det/G_randの定義とRPE総費用式を確認。Q81の式はE.3の総ゲート式であり、元論文の分子数値・Toffoli表を今回のRZ値へ変換したものではない。元論文のN_mの係数を今回のN_aの式として転記しない。
- 固定commit `427480e4f1a61385cbffa724930a2221ed9e5e1d`：`docs/research/track_a_h46_theory_compiled_paper_evidence_20261018.md` と `artifacts/resource_applicability/track_a_h46_theory_compiled_paper_evidence/2026-10-18/table_index.md`。モデル／native／compiledの同一event計数、S4有限精度修正、loweringとopt1の未分離を照合。
- 固定commit `da8eb64ba8e88c219a6b707e1c72d130f5f37fcb`：`scripts/h46_cost_summary.py` の保存shots利用・`no_new_bias_or_shot_rule=True` と総費用の合成。現行スライド6の図はN固定のC比較で、独立N比較ではない。現行発表用6・詳細A.9の条件と旧n8を保持。

### 新しいC・N・G比較資料（今回の本編へは未採用）

固定commit：`b81d4096e806efb6523304764d3a0680b7663320`。
branch：`codex/pr-pf-analytic-vs-circuit-cng-20261022`。GitHubのbranch headをread-onlyで照合した。この枝を発表資料枝へmerge/cherry-pickしていない。

root：`docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/`。

|固定ファイル・表示群|確認事項|
|---|---|
|[README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b81d4096e806efb6523304764d3a0680b7663320/docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/README.md)、`definitions.md`、`captions_ja.md`|C/Gの20条件、native期待RZ数と保存compiled値、G00→G01はN固定。新図の本編採用は未判断|
|[N_comparability_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b81d4096e806efb6523304764d3a0680b7663320/docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/N_comparability_audit.json)|`independent_conditions=0`、`required_conditions=20`、`N_circuit=null`、`N_ratio=null`、`zero_N_change_observed=false`。新推定器・分散適応は未採用|
|`coverage_matrix.md`、`figure_data.json`、`provenance.json`|直鎖1 Å・STO-3G・DF3/10/19/28・T=.8・生成順・通常signed・倍率1・P0・5精度、固定したS2 PR/PF候補、δ=.8/q、Qiskit1.3.0/Nature0.7.2・rz/sx/x/cx・opt1・seed17・topologyなし。H2は元n8、他3系は新cohort。H6/H8は補助演算子unweighted-G診断不合格の条件付き参考値|
|`confirmed_H2_H4/` と `all_conditional/` のC/N/G図|前者は実施済み技術検査PASS範囲、後者は条件付きH6/H8も表示。独立科学レビュー済み・独立確認Pとは呼ばない。N図は保存回数の情報図|
|`N_comparison_contract_DRAFT.md`|未採用・未実行。今回の読解・Q&A追記を、新しいN比較計算の承認へ拡張しない|

元SOURCEは `bb5251fc9780ac7ac3b6c3c540aa3c865879eb4f`。固定README末尾の「ローカルcommit・pushなし」はその作成時の記録で、今回確認したbranchの公開状態とは区別する。今回は既存blobの読解と表示確認のみで、科学計算や新しい科学的判定は行っていない。

### outer32と発表用の表示改善

- outer32は既存 `figures/all_methods_hchain_with_higher_pr_20261022/README.md`、caption・provenance、詳細B.12、本編10の条件を参照。旧主比較とのidentity結合未実施を維持し、Q91のT=.8/q4は記号説明例としてのみ記す。
- 本編1・13の表示改善の公開基点は `9b042caa3eaebcf023b3e74e8327990368a6e6e4`。[表示改善記録](2026-10-22_PR資源評価_発表用補足8枚_Codex参照メモ.md)と[その時点の文書検証](2026-10-22_PR資源評価_発表用補足8枚_文書検証.json)も参照。Q39は本編1、Q94は本編13/S5の既存表示の転記。今回数値・保存判定を変更していない。

## 旧記録の扱い

以下は2026-10-21のQ&A作成時の78件・参照先・未公開状態の記録。その後の発表用別版・公開・今回の96件への更新を取り消す指示ではない。

<details>
<summary>2026-10-21作成時の記録（原文保持）</summary>

作成日：2026-10-21 JST。対象は[Q&Aノート](PR資源評価_スライドQAノート_2026-10-21.md)。本メモは参照・編集継続用で、投影用ではない。

## 作業範囲

これまでの会話のうち、スライドの内容・結果の読み方・用語・構成に関する質問を78件に整理した。繰り返しや表記の質問をまとめ、途中で限定・訂正した説明は現行の理解へそろえた。これは会話と保存資料の編集であり、詳細科学レビューや新規科学解析ではない。

基準文書は[説明意図明確化版](PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_説明意図明確化版.md)。この基準文書・過去版・既存メモ・図・結果を変更しない。案内READMEにはQ&Aの入口のみ追記する。

## 質問と確認箇所

|質問|確認箇所|
|---|---|
|Q01–Q08|構成案の本編1・4・5、B.1/B.8/B.16/B.18。rは `src/trotterlib/rte.py` の `RTEConfig` と `df_partial_s2.py` の config.evolution_time＝outer step_time照合を読み、δ/rの内側分割であることを確認|
|Q02|`src/trotterlib/df_hamiltonian.py` の `df_hamiltonian_from_integrals` / `_low_rank_kwargs`、OpenFermion `low_rank_two_body_decomposition` の final_rankとtruncation_thresholdの分岐。個別の係数閾値と累積打切り基準を混同しない|
|Q09–Q20|本編4・5・8–10・13、B.3/B.13/B.16/B.18。B.16にH4既存1候補n32＋新49候補n8、図の採用PRはすべてn8と明記。H2粗精度のn80はA1左P|
|Q21–Q28|本編2・3・6・7・11、A.1–A.9、B.2/B.6/B.7/B.16/B.17。実compile、固定モデル、有限標本平均、保存shotとの積を区別|
|Q29–Q35|B.4のcommon/prefix/Tail、full-Fock identity、scalar一回、参照不回転。Q&Aは同節の記号をc_Rへそろえた説明転記|
|Q36–Q39|本編1、B.5。arcsinの上界とsinの要件換算を、単一固有状態・半径・branch等の条件付き関係として説明。新しいHa表は算出しない|
|Q40–Q43|B.11/B.14。総α変更とα配分、旧RPEのβ配分、Re/Imへのε配分を区別。旧H6/H8 n32と現行Pを混ぜない|
|Q44–Q51|B.15/B.19/B.23、前版PF費用変化量メモ、保存100点。signedと経験的非相殺、旧70/新30、scopeとendpoint、政策間最小費用と候補切替の寄与を区別|
|Q52–Q59|本編5・6・8–11、A.8/A.9、B.10/B.12/B.13/B.16/B.17。原論文の全資源式との直接検証済みとは書かず、有限RTEの理論期待費用を具体的DF回路へ接続する既存結果はあると説明|
|Q53の標準S4|固定SOURCE `c0d8ca8ecd3fc29528ca8424f8b8d953d06eff91` の `src/trotterlib/native_df_gate_model.py` にglobal Yoshida S4 scheduleの記述。`product_formula.py` の `yoshida_4th_list` と標準4th分岐も静的確認。new_2元8桁版を同列の認定済み四次とは呼ばない|
|Q60–Q68|本編12、B.20、説明意図明確化メモ。C1/C2は既存保存分布による表示で、新bootstrapなし。青帯の分位区間、青の平均、保存順の緑、灰の全保存平均を区別|
|Q69–Q78|説明意図明確化版0/15/19、B.21/B.22、各版の参照メモ・README。旧スライド番号を現在番号へ対応させる。公開・local commit・ローカル保存を区別|

## 固定SOURCEへの入口

- [説明意図明確化版の参照メモ](2026-10-21_PR資源評価_説明意図明確化版_Codex参照メモ.md)：今回の基準版と前版7c0396beへの対応。
- [PF総費用変化量の参照メモ](2026-10-21_PR資源評価_PF総費用変化量_Codex参照メモ.md)：結果3983fe4f、科学SOURCE c0d8ca8e、保存100点、旧70/新30、H8の2 scope。
- [図・qモデル検証更新版の参照メモ](2026-10-21_PR資源評価_図・qモデル検証更新版_Codex参照メモ.md)：簡潔版6図、450軸E–E4、H2受入、旧来歴。
- [旧構成案の参照メモ](2026-10-18_PR資源評価_スライド構成案_Codex参照メモ.md)：旧H4/H6/H8、α・β、高次PR、理論／compiled等の固定資料への詳細対応。履歴部分の当時の枚数・未完了状態を現在の状態としない。

## 訂正・限定を維持する点

1. 「探索Aはすべてn8」は掲載された現行H-chain図のPR点に限定。全登録候補へ一般化しない。
2. 独立確認は候補・対照を固定した別標本。候補探索数の拡張でも、探索標本とのpoolingでもない。
3. rは1つのTail区間の内側RTE小ステップ数。nと取り違えない。
4. H2の独立確認2条件は有限標本結果として受入済み。主表追加は未決定。
5. B0はHamiltonian全体のIを落とさない。B0+Iはdiscarded Tail由来の既知Iだけを回収。
6. 「signedと非相殺のどちらも含む比較条件」を、会計を混ぜた単一比較としない。
7. bootstrap平均の分位帯を母平均の厳密CIとしない。H4からH8の数％差や必要nの十分性を認定しない。
8. 青線・緑線を本編から外す案は提案のみ。図は実際には変更していない。
9. 理論／compiledの旧H6図はPF＋PR2候補の3パネルであり、P/A図とは異なる。
10. 候補集合拡張の履歴は保持するが、最初から全範囲が事前登録されたとは説明しない。

## 文書確認と終了

Q01–Q78の連番・重複・内部anchor、目次、14枚／12枚の対応、ローカルリンク65件、Markdownの表・数式を確認済み。既存発表資料129ファイルは、案内READMEへの追記を除きSHAで保全を確認した。元の構成案・メモ・全図は不変。主要な保存表示値も構成案の既存記載と照合し、再計算はしていない。

2026-10-21のQ&A作成時点の基準構成案のSHA-256は `f5b9d4bb4d4109933e39b456410e57848dc46e920b68968e7474d51a9251c7db`。技術確認の実行記録は作業用一時ディレクトリに保存し、新たな科学台帳・検証結果として登録しない。

新規科学計算・sampling・compile・統計再解析・bootstrap・費用再計算・再ランキング・図変更は行わない。既存SOURCE・原結果・主表・grant・ledger・marker・STOPは保全する。Q&Aと本メモの作成、案内リンク、文書確認までで終了し、commit・pushは行わない。

</details>

## 2026-10-22今回の文書確認

Q01–Q96の連番とanchor、既存78見出しの保持、18件の追加、14枚／12枚とS1–S8への対応を確認した。ローカルリンク100件が解決し、数式33箇所をMarkdownの数式として解析できることと括弧の対応を確認。Ha表は本編1、探索→確認表は本編13/S5の保存表示と照合し、新しい算術・統計計算はしていない。CNG固定資料の `independent_conditions=0`・N関連null・`zero_N_change_observed=false` も照合した。

作業は公開用の既存worktreeで3文書だけに限定した。共有checkoutの発表資料503ファイルは編集前SHAと照合し、今回の3文書を除く発表用本編・詳細版・全図等を保全している。技術確認記録は作業用一時ディレクトリに保存し、新しい科学台帳や科学検証結果として登録しない。ユーザーの継続指示に基づき、文書確認後に3文書をcommit・pushする。公開commitはGit履歴で追跡し、本文へ自己参照SHAを埋め込まない。

新規科学計算・sampling・compile・bootstrap・統計再解析・費用再計算・再ランキング・図変更・原結果変更・詳細科学レビュー・最終evidence freezeは行っていない。新CNG図の本編採用や追加N検証も実施していない。今回の文書公開後に残る必須作業はなく、mandatory STOPとする。
