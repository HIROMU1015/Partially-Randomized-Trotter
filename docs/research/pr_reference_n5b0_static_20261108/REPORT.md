# 新A N5-B0：m17 precision・emitter・域の静的実行可能性調査

2026-11-08 JST、SERVER 2。固定基準 **d1d5992baa6b818fb17fb0c05ba0c100bab3806d**。Branch `codex/pr-reference-n5b0-static-20261108`。

**静的調査を完了しmandatory STOP。現在の保存certificate／providerをそのままm17へ移す資格はないが、実装の不可能性も新規合成の必然性も証明していない。** 問題は保守的emitter上界だけでなく、logical guard、b36量子化、域の組合せにある。新しい小さい上界、precision、emitter、辞書、policyを採択していない。

## 結論

| 論点 | 確認できたこと | 未確定事項 |
|---|---|---|
| C0の64u | SOURCE由来の一様な安全上界。実測値・下限・最適値ではない | 同じword／emitterで全位置のstrict上界をどこまで鋭くできるか |
| W residue | 全8 residueの保存wordあり。Local loopの8状態閉包を再利用できる | Local residueだけでは非zero prefixとのjoin・shellを覆わない |
| 共通joint scalar | 固定consumerはnamed gate列とnumeric global_phaseを分離し、合成後のcontrolを加えない | Scalar因子分離をRPE信号保証へ正式に接続するための証明・適用確認。今回は未採択 |
| Logical guard | C0 libraryの11 wordはguard成分だけならm17両armのY cap内 | それだけでemitted certificateやjoint成功が資格されるわけではない |
| 基本word | 実guard照合ではB′ cap内0/43、Reference cap内1/43 | 選択chunk和・量子化を含むm17保証。Precisionラベルだけで再利用可否を決めない |
| b36と域 | 保存量子化上界はm7用。固定Lはm17包絡を覆わず、B′ scalarの保存literalも域外 | 必要target集合に適合する保証／構成。新b/j範囲・縮約方法は未選択 |

優先する次の技術候補は、**新規合成前に、同じword・同じconsumerのphase因子分離とstrict評価の保守性を限定検証すること**。それが成功してもlogical guard／量子化／域の不足は残り、m17のRPE PASSを意味しない。

## 固定条件と保存RPE条件

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、既存D融合・basis共有・X/Yを維持する。

M17は **q131072／r32／88 bits、B′626・Reference630 shots/axis**。保存gate十分条件Yは **B′6.902168833913054e-16、Reference2.5648979454057876e-15**。Energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720も変更しない。これらは未加算CT誤差に対する保存条件の引用であり、新しい必須precisionや不可能性境界ではない。

M7の199-shot PASS、strict10^-11、b36、L=82.49247654978845をm17へ採択していない。[N4-C](../pr_reference_n4c_accounting_20261108/REPORT.md)のm7結果と旧STOPはそのまま保全する。

## 成果物

- [M17_ERROR_BUDGET_BLOCKERS](M17_ERROR_BUDGET_BLOCKERS.md)：target生成／量子化／guard／W具象化／emitter／shell／旧diskの帰属、保存RPE入力。
- [EMITTER_BOUND_OPTIONS](EMITTER_BOUND_OPTIONS.md)：64uの保守性、局所residue評価、共通joint scalarの条件付き代数、必要証拠、adapterの要否。
- [ANGLE_DOMAIN_AND_WORD_REUSE](ANGLE_DOMAIN_AND_WORD_REUSE.md)：m17のsigned scalar／event／r32中央／vacuum、再利用と域外の分類。
- [NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)：凍結した既存wordだけを使う最小の後続技術検証案。未実行・別途承認が必要。

[WORD_EVIDENCE](../../../artifacts/pr_reference_n5b0_static/2026-11-08/WORD_EVIDENCE.json)は299 C0・43 basic・85 certificateの**既存guard等の引用とmetadata比較**。新guard、error和、phase/radius、shot、価格の計算ではない。Logical guardだけの適合件数はlibrary件数で、m17の物理位置数・全必要word資格ではない。Guard上端がcapを超えることも、実誤差の下界にはならない。

## 静的調査の実施と限界

[Helper](../../../scripts/audit_pr_reference_n5b0_static.py)はCT-0 positive／validate_recordのmetadata関数だけを変更せず再利用した。保存archiveのhash・spec・sequence・Wの対応と、既存guard/capの大小を照合。W mod8はinteger metadataであり、RZ角をmodで置き換えたものではない。

Strict_guard／exact_word_matrix、lower_native、provider、consumer、CT-0／CT-4／N4-Cの数値会計SOURCEは読取のみ。Qiskitのinstalled quantumcircuit.pyは、N3-B1aに保存された公式commit対応のbytesと一致した。Compose／inverse／global_phaseのデータ経路を静的に追っただけで、新runtime数値検査は起動していない。

[SOURCE_REFERENCES](../../../artifacts/pr_reference_n5b0_static/2026-11-08/SOURCE_REFERENCES.json)、[M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)、[CHECKS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/CHECKS.json)に保存した。正式共通運用指示2026-10-25、親／worktree AGENTS、PROJECT_MAP、既存研究概要・固定資料を適用した。著者Codexの静的技術分析であり、独立詳細科学レビューや正式な保証採択ではない。

統括兼Dのnative CX/RZ/depth、q64検証、13-round adapter、回路再設計は再実施していない。未取得の他チャット成果を推測で補っていない。小さい読取処理で独自CPU/RAM capや共有設定変更はなく、他ユーザー・他Trackへ干渉していない。

**新規科学数値再解析、guard、合成、provider/emitter実行、回路build/compile、simulation、sampling、RPE再会計、価格集計は0。** 旧SOURCE／raw／result／grant／ledger／marker／STOPを変更していない。

**[MANDATORY STOP](../../../artifacts/pr_reference_n5b0_static/2026-11-08/STOP.json)。次担当はGPT／ユーザー**：保証経路と後続限定検証の採否を判断する。新しい保証の採択と数値実行の承認を区別し、CodexはN5-B1／新emitter／新規合成／m17再会計へ自動進行しない。
