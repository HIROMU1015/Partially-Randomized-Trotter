# 2026-10-13 mixed-access feasibility v1 一回実行

[一次結果](../qc_mixed_access_feasibility_v1_results_20261013.md)、[GPTへの引継ぎ](../qc_mixed_access_feasibility_v1_review_handoff_20261013.md)。固定scope説明後のユーザー「次の作業に進んで大丈夫」をH4一回の明示実行承認として記録し、214f9ab2でpreflightとともに固定。102e6b31のscience sourceは変更しない。

H4直線1Å/STO-3G/実RHF8mode・ν4、16DF全保持、L_Dなし/PRなし、T1/4・r1…1024倍増、εU=.001・7法154queriesを一回で完了。SCF通常5iteration、ground/gap/FCI0。保存監査全154 PASS、1386resource rows/273frontiers照合PASS。登録template/κhypothesisのremaining budgetは全て負で、NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL。

ideal step改善は有限gridで確認できたが、full quantum cost優位や一般的不能は認定しない。新科学結果の解釈・投資判断はGPTへ戻し、開始承認を待つ。Codexはbackend・追加scan・別molecule・別Trackを自動で起動しない。N1STOP/N2HOLD/N3CLOSEDと全旧結果・認可・原契約・STOP、global既存descriptionFAILを保持。Mandatory STOP。

公開追記：data c23b3f14の保存logがignoreで未収録だったことをremote監査で検出し、ae3975deへ保存済みlogを追加。原結果/source/raw/索引不変、追加科学計算なし。証拠payload ae3975de24bff6ab6aec3494b522851db2a8f3ddをGitHubから再取得し、196blobs/65,061,432bytes、生成182/raw163/source16/旧12/links48一致。[引継ぎ](../qc_mixed_access_feasibility_v1_review_handoff_20261013.md)。科学判断開始承認待ち、Mandatory STOP。
