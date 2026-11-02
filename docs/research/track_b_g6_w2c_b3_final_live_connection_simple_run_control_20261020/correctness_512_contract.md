# 512件correctness契約候補

[manifest候補](correctness_run_manifest_candidate.json)は未承認テンプレート。予定Run IDと新しいoutput/state pathを含み、directory・real permission・claimは作成していない。
4方式×A/B×X/Y=16cell、各16trials、S/P1それぞれ256、合計512。
correctness/baselineだけ、profile/instrumented pass/追加sampling0。S→P1、単一global one-shot、retry0/resume=false。

Primary16、G7make14、F3.prepare2、AuditParity8、canonical snapshot6。
B sampling audit/angle/requery各最大256、Canonical angle192、Full angle64、correctness event_witness512。
Full sample packet128、Full angle packet64。F3.prepare内部constructor2はprimary16の内数。
Nested B witnessとdriver correctness witnessを分け、同名関数の総call上限512とはしない。
3200draws案はCanonical384×6＋Full128×7、固定m5・raw degree≤4・H160・audit/query追加draw0・retry0に依存する。
内部root/law/predicate等の総call数は未確立。[別予算](call_budget_manifest.json)。

Science seed候補 `PROPOSED_B3_CORRECTNESS_ONLY_20261020_V1`、SHA256 counter方式、domain=namespace/arm/axis。
modeはglobal trial identityに含め、seed domainに含めない。A/B・S/P1の対応prefixを比較する。
各cell accepted≥1を維持し、16trials完了で0ならINCONCLUSIVE_COVERAGE。延長・retryなし。
科学不一致・技術STOP・証拠不完全を別分類にする。
保存はS/P1共通result1GiB/log64MiB、全512 full Evidence。超過時はSTOPし間引かない。
実science Evidenceの容量十分性、accepted/reject分布、IIDはNOT_ESTABLISHED。


**実科学未実行、実行未承認。mandatory STOP。次の担当：ユーザー・GPT。**
