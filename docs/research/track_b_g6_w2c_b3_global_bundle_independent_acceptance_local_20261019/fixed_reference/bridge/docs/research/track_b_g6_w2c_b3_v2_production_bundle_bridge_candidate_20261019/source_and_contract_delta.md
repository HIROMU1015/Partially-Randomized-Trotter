# 固定SOURCEとの差分

|役割|固定点／新版|
|---|---|
|直接基点Producer|596781daeba253418ba35f4b7964312a54caf233|
|Producer人工測定SOURCE|45bf46b4a833e71f138903fb511f2741e54043b7|
|主要固定受入|77126767c00585d0967558c0756907df2d69d1ff|
|旧独立core/test|49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866|
|今回の人工runtime測定SOURCE|e099dcc461247e81eff981867dc28c3f3b16df5e|

101公開ファイルのbytesをGit blobへ照合し、測定45bfは公開履歴bundleからfresh bare Gitへ復元した。bundle 49,475B、前提76ca3c40、15測定SOURCEの全bytes一致。公開HEAD後追加helper2件とは区別する。

新規6moduleはbundle contract/bridge/fixture/store/trace/offline v1。旧Producer・独立checker・G7/F3・R1/F01/Admission/Hostの既存bytesを編集していない。新schemaとrunner/testsを追加し、branch間の無条件merge/rebaseは0。

実質技術差分は、単一global台帳への二条件結合、modeを含むglobal trial ID、6phase barrier、dispatch中断PHASE_ABORT、post-close global receipt、receipt close後のmarker、callerによるdelivery close観測、最後の証拠slotの事前global creditである。旧V2のmarkerがcloseより前に残り得る履歴は修正しない。新順序と外部観測joinの正式採択はユーザー/GPT判断。

新人工endpointは人工専用surrogateであり、law/weight/evidenceの人工形状は旧fixtureとの科学的同値を主張しない。人工SHA256 counter namespace・160bit消費・S/P1/A/B/pass prefix対応を維持し、科学law/event/precision/angle参照は固定bytesとして保持した。実科学APIの同値実行証明は今回取得していない。

測定後のtest修正は、出力前Admission拒否の期待分類とmutation証拠overlay保存だけ。新tests/evidence helperの現行SHAと `at_measured_SOURCE` をinventoryで区別する。runtime six modulesとrun_case/source/schema pinsは測定後変更していない。
