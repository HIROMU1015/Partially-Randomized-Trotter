# 第一候補・H-E1：既存成果の簡潔な引継ぎ

2026-10-15 JST。ユーザー提供の[完了GPT科学レビュー](../reviews/quantum_chemistry_pf/qc_post_he1_research_direction_scientific_review_20261015.md)を原文保存し、今回のAとして既存成果だけを整理した。論文草稿、再集計、科学レビューの再実施ではない。

| 系列 | 目的・確認済み成果 | 現構成を区切るレビュー理由 | 未解決事項・正式状態 |
| --- | --- | --- | --- |
| 第一候補：一体項再配分PF | 元Hを保つfirst-pair再配分とexact-ground主要誤差の1-RDM応答。固定H4で主要係数・有限h挙動の整合を確認した | 同情報compact BCHに応答生成の独占的演算量差なし。主要係数の最小化はCS2補正後残差を直接狙わず、登録10%費用改善に未達 | 情報取得・controlled合成・全QPE/RPE費用・他分子移送は未確定。終了推奨 `CLOSE_CURRENT_CONFIGURATION_RECOMMENDED`、正式状態 `AWAITING_USER_ADOPTION_DECISION` |
| H-E1 | PF構成と補正情報取得の共同費用という仮説をK0で既知文献・第1/第2研究と比較。具体化可能な処理系は `REDUCES_TO_KNOWN`、仮説全体は `UNDETERMINED` | 独立した生成規則 `NOT_IDENTIFIED`、`gate_pass=false`。今回のGPTレビューは現定義を独立算法研究として追究しないことを推奨 | K1〜K3未実施。copy-only／準備unitary付きaccess、測定分散・精度・情報価格を区別。第1/第2研究へ自動移管せず、正式な新研究採択なし |

第一候補の数値的記述は、linear H4・1.00 Å・STO-3G・保存DF rank12 generation順全12 density、Nα=Nβ=2・36次元、固定first-pair一方向X、h=0.002745119306432635〜1 Ha⁻¹の381比較点（補助含762）、target10⁻⁴/10⁻⁵ Ha・P1/P2というdevelopment scope。PRのL_D/delta探索ではなく、static controlled primitive／Heisenberg proxy・保存gridの事後比較である。今回は数値を再計算・再ランキングしていない。

原証拠と詳細は以下へ残す。

- [第一候補の固定結果・独立監査への索引（500ea6）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)
- [第一候補の統合完成版レビュー（0e923f）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/reviews/quantum_chemistry_pf/qc_onebody_repartition_h_boundary_final_scientific_review_20261015.md)
- [H-E1 K0監査・固定文献／参照索引（0e923f）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/research/qc_he1_stage0_technical_prevalidation_20261015.md)
- [今回受領レビューの原文identity・著者来歴](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/review_identity.json)：42,646 bytes、SHA256 `326622956c9eca9d28fc8cc9d500ad3cf1d45e7b2809ab06eea23aac1c750fca`。レビューが参照するClaude原文のSHAは同レビューによる報告値として分離し、Codexが原文を独立取得・計算したとは記さない。
- [後続M案K0の技術監査](qc_m_fixed_circuit_k0_technical_audit_20261015.md)。Aの整理完了後、今回許されたBの静的監査だけを実施する。

一体再配分・H-E1一般のno-go、情報価格込みの資源優位、全ground-energy taskのGOは確立していない。第一候補の正式CLOSED変更、既存系列の再開・移管は行わない。旧結果・source・消費済み認可・Mandatory STOP・manifest既存FAIL、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template CLOSED、C1〜C4・Track A/Bを保全する。今回の新科学計算・保存数値再集計は0。
