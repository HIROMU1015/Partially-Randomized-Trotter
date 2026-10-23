# Phase 2D-F1-PRE

状態：**READY_FOR_F1_EXECUTION_APPROVAL**。専用環境・保存B入力への接続SOURCE・case ID計画を準備した。ユーザーが追加した正式共通指示 `research_project_instructions_unified_2026-10-23.md` を読み、§8.4・§10（§10.3を含む）の再利用・資源・研究目的に応じた管理方針を照合した。旧版で代用していない。

下記の価格／interval境界を**次の実行承認の候補**として明示した。まだP0の承認条件を変更・採択していない。READYはこの具体的な範囲を承認判断できるという技術状態であり、full P0価格資格済みという意味ではない。

- literal recipeの数値価格bindingは研究回路生成を必要とするため、F1承認後の登録price case内へ置く。
- interval providerは未資格のまま。4armのliteral期待価格とcatalog直接列挙／DPは検査する。selectorと全列挙は既定の `UNPRICED_FALLBACK` を記録し、認証score・shot数・価格による優位選択を報告しない。この境界を承認せず、P0の価格事前binding／資格条件を要求する場合は別の準備範囲が必要。

専用CPython 3.11.9 venv、NumPy 1.26.4、SciPy 1.14.1、Qiskit 1.3.0、mpmath 1.3.0を用意。依存50wheelのhash lock、通常import、API signature、依存整合、管理テスト2件、静的検査を確認した。**科学的PASSは取得していない。**

新しいHamiltonian／event／trajectory／研究回路／compile／科学test・runnerは実行0件。grant発行・消費0件。B原本40fileと固定B環境をread-only照合し、入力の再生成・数値再解析はしていない。

入口は [技術記録](TECHNICAL_PREPARATION.md)、[固定identityと承認候補](manifest.json)、[依存lock](requirements.lock)、[引継ぎとSTOP](handoff.md)。旧prototype・旧runner・旧20testは保持し、追加は保存入力adapterと管理テストのみ。新SOURCEの科学検証はF1承認後に行う。

GitHub公開・照合後はmandatory STOP。実行範囲の承認判断はユーザー、最終packet準備と承認されたF1実行はその明示指示を受けたCodexが担当する。F2と詳細科学レビューは自動開始しない。
