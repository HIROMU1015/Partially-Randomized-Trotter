# H4 Pauli-JW／DF-native PF・PR比較の再整理

[技術報告書](../../../docs/non_df_h4_comparison_synthesis_20261025.md)。固定H4/1Å/STO-3G、Pauli184/prefix27、DF tol-only1e-8/rank10/prefix1＋one-body、S2/T.8/q8/r4/K4。

- `circuit_costs_QUOTED.csv`：generic Pauli・central Pauli・DF native各PF/PR、6系列の保存RZ価格。PRはn=256標本平均。
- `conditional_resources_QUOTED.csv`：generic/DF20＋central10の仮想ρ=0条件のN/G。元JSONのpointer付きで、再会計ではない。
- `evidence_quotes.json`：入力、model metadata、D1初回停止とSOURCE基準Bの9測定、D2、bias/normalization/guard、希少event、6局所Arb、保存比較領域の引用。
- `verification.json`：引用の同値・SHA、旧SOURCE/manifest/証拠の保全、文書リンクの構造確認。
- `publication_status.json`：commit/push・GitHub remote SHA確認結果。

現在のempirical/conditional case studyを報告する証拠は揃う。無条件の精度保証付きPR母平均G優位は未確定。
D1初回の旧Pauli-PR hash FAILは維持するが、後続の別SOURCE基準では保持/event/emptyを含む9構成の価格が測定済みである。
未測定のbasis単独価格・pass寄与率を補完しない。6局所保証をPR全law/DFへ転用しない。
新H4科学計算・数値再解析・再ランキング・build/compile/sampling・R2-C証明継続・正式受理は0。
研究方針の採択はユーザー/GPTへ返し、今回の保存・公開照合後mandatory STOP。
