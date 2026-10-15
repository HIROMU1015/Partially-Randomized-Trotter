# SOS-PF候補A：K0監査packet

## 公開用注記（資料公開のみ）

この版は、ユーザーが別途承認したK0資料の限定公開用である。以下と各成果本文にある「ローカル資料のみ」「未公開」「commit/push未認可・0件」は**K0終了時点の履歴**であり、今回の公開許可と区別する。科学的内容、UNDETERMINED_DATA_REQUIRED、K1未認可、正式採択なし、Claude原本未取得、Mandatory STOPを保持する。

公開対象はこのディレクトリの10ファイル。技術監査・文献台帳・method status・GPT引継ぎ・K1草案・受領指示・指定GPTレビューはK0原文とbyte同一。READMEへの本注記、provenanceの固定URL追加とローカル絶対パス除去、保全確認の公開範囲注記だけを整理した。元のローカル成果は変更していない。無関係なworkspace全体のファイル名・hash一覧 preservation_baseline.json は公開せず、ローカルに保持する。公開前のmetadata hashはprovenance／保全確認に残す。

arXiv:2510.06316v3との対応は[技術監査 §2](technical_audit.md#2-k0-1先行研究と新規性)、[文献台帳 L9](literature_registry.json)、[GPT引継ぎの近接先行研究](gpt_handoff.md#追加の近接先行研究)にある。共通部分、access modelの差、未確認の有限step保証・独立生成則・総費用を区別している。本文の追加研究や新しい文献調査は行っていない。

既存source・採用decision・過去の監査は[provenanceの固定commit URL](provenance.json)から取得する。研究状態の[既存採用decision](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json)と[M案K0原固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_m_fixed_circuit_k0_technical_audit_20261015.md)は再複製していない。

今回の認可はcommit/push・再取得照合に限定し、GPT科学レビュー開始、K1、追加計算を含まない。

2026-10-15 JST。**K0完了／全体 `UNDETERMINED_DATA_REQUIRED`／K1未認可／正式採択なし／Mandatory STOP。** ローカル資料のみ。既存科学source・結果・認可・STOP・manifestを変更せず、新しい科学計算とcommit/pushは行っていない。

| 資料 | 内容 |
|---|---|
| [技術監査](technical_audit.md) | K0-1〜K0-5、独立導出、先行性、表現、費用、停止理由 |
| [文献registry](literature_registry.json) | 固定版URL、確認節、仮定、保証・費用、取得限界 |
| [Method status](method_status.json) | 項目別評価と実行・採択境界 |
| [GPT引継ぎ](gpt_handoff.md) | 重要結論、残った新規性、次の科学判断 |
| [K1条件付き草案](k1_design_draft.md) | 未採用・未認可。中心生成則が未確定で実行契約ではない |
| [受領指示](received_instruction.txt)／[指定GPTレビュー](supplied_gpt_review.md) | 原文byte保存。レビューは今回の成果ではなく入力 |
| [Provenance](provenance.json) | 固定Git object、入力SHA、取得範囲、未取得資料 |
| [保全確認](preservation_check.json) | K0時点のbyte保全・文書metadata検査。workspace全体のbaselineはローカル保持 |

数学の中心とPF回路への接続は成立するが、有利な独立生成則・実用的有限step保証・同情報の強対照に対する総費用利益は未確認。既知のPSD化やSOSとPFの組合せだけを新算法としない。一般的不能性も認定しない。

既存の研究状態を更新する資料ではなく、今回のみ許可された計算前監査の新規packetである。既存索引・概要・manifestは今回の変更禁止範囲として保持した。入口は本READMEとGPT引継ぎ。
