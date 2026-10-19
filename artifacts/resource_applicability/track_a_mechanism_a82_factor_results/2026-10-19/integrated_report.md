# A82比較説明因子のone-shot完了・Mechanism統合

**A82_OLD_PR一比較のcosine/sineをone-shot1回で正常完了。固定32/32比較の説明因子64軸が揃い、保存結果を論文資料へ統合してmandatory STOP。**

SOURCE `06c94745f50345b2d93714760608d9d11b83e784`。準備commit bfa47ba92925c16bcb99001fb28974ddaef57cdaの固定契約と入力を変更せず、直前の承認依頼に対する利用者の「次の作業に進んで大丈夫」を当該one-shot実行承認として新receipt/grantに結合した。旧未承認契約表示・無効grant・STOPは履歴として保全。

H8 linear1Å/STO-3G/tol-only DF28/T=.8/P=0、通常signed/factor1、ε=.05、PR B2_p7_q2_2nd_r4_K2対PF B0I_p11_q1_2nd、旧confirmation union n32。検証済み保存PR/PFから各軸7因子/7logだけを算出し、source/input/runtime/controlを実行前後に照合した。親/workerともnative exit0・wait/reap、保全381 fileがSHA一致。retry0。新sampling/compile/signal/統計/ランキング0、core32・side・他31比較の再計算0。

[A82の出力delta](A82_factor_delta.json)と[因子値](A82_factor_values.md)、[全64軸因子表](data/explanation_factors.csv)、[主表32行](data/main_table.csv)、[status32件](data/comparison_status.csv)、[論文manifest](paper_manifest.json)、[Claim–Evidence更新案](claim_evidence_matrix_update_DRAFT.md)を保存した。元62因子行はbyte prefixとして保持し、新2行だけを追加。主表の元数値・元status列はそのまま、新しい完了status列を追記した。旧欠測や31/32の資料は上書きしていない。

[Figure 2補遺](Figure2_caption_addendum.md)と[来歴](figure2_provenance_addendum.json)を追記。既存SVG/PNG/PDF・116点・全数値・cohortは不変で、A82因子の追加描画はしていない。H4独立Pはlegacy DF12の別cohortのまま。H6直接実測/model、H8独立n48と旧A82 n32の区別を維持。

固定primary32の説明入力・比較因子欠測は解消。母平均優位・未観測rare event・厳密bias/電子Hamiltonian誤差・PF model範囲・FT資源等の従来制約は残る。[evidence readiness](paper_evidence_readiness.md)。この統合は主担当のlocal技術照合で、詳細科学レビュー・独立review・最終freezeではない。H4/H6並行検証・共通protocol v1・共有環境・他jobへの変更0。

新成果を独立local commitに保存する。公開承認の範囲は未確定のためpushなし、remote SHA照合は対象なし。[公開状態](publication_status.json)。次担当はユーザー：公開範囲と、必要なら図の表示更新を判断する。固定32比較の因子を完成させるための追加科学解析は不要。[次作業](next_actions.json)。
