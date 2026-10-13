# THRIFT型mixed-access feasibility：次の実行承認scope

2026-10-13 JST。[固定計画](qc_mixed_access_feasibility_plan_v1_20261013.md)、[機械可読契約](../../artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json)。**これは承認依頼の対象範囲の準備資料で、実行認可書ではない。**

完了したGPT判断を反映：現constructorを独立新規算法として非採択。既知mixed-accessで追加算術を回収する費用の余白を調べる。新規性の再監査・GPTレビューを開始しない。

将来の一回検証で認可する範囲：

- H4直線1.00Å/STO-3G/実RHF/8spin modes/ν4/frozen coreなしの一input生成。FCI・多体ground/gap eigは0。SCFの一体対角化とDF小行列eighはscopeに含む。
- P2、excitation E2/E4、DF D2/D4、known-compressed THRIFT R2/R4の7法。T=1/4、r=1…1024のdoubling11点、154以下のideal error probes。full-Fock256の出力からν4の70input columnsを比較し、Pauliのsector leakageを含める。
- model/precisionを揃えたpartial resource ledger、function remaining費用のsymbolic ceiling、κ=3/6/12の明示hypothesis sensitivity。compile・T synthesis・function backendは0。
- 2workers/thread1、RSS合計4GiB、output64MiB/log4MiB、attempt1/retry0、wall/CPU capなし。結果・失敗・独立監査を保存・公開してMandatory STOP。

実行前には通常のrunner/checker・新pipelineの対象外tests、dependency lockとsource SHA固定が必要。現planのrunner/input SHAはnull、PySCFは現venvに未導入。承認後もlaunch gateが揃うまで科学実行しない。これらの技術準備だけでGPT再レビューを要求しない。
旧H2/H4案を起動しない。H2・別geometry・追加r/精度・ground-energy/QPE・PR・N1/N2/N3再開・第二候補への移行は含まない。
positive budgetは実backendがその費用に収まる証明ではない。negative budgetも指定model/有限scopeの結果。結果取得後の研究継続・詳細backend投資判断はGPTへ戻す。

現時点：計画作成・保存のみ、科学計算0、実行承認待ち、Mandatory STOP。
