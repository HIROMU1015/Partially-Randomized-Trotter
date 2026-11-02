# Track A V2：保存済み理論評価との読取専用照合

V2のprimitive生成と基本A/B表の再計算は重複するため実施しない。
利用者指定の `track-a-native-df-theory-evaluation-20261016`、結果commit
`7e41670565d9284da9aded94063be591fdcaa3dd` を、V2準備commit
`dd695e6584f15b88341d2446ed010c7bdfef1666` の計画と照合した。
この文書と照合receiptはV1準備branchに保存し、V2・既存評価branchには書き込まない。

[静的照合receipt](../../artifacts/resource_applicability/track_a_v2_saved_theory_reconciliation/2026-10-16/reconciliation_v1.json)。
資料の読取り、件数・候補ID・hash・保存状態の照合のみ。数値counter、producer、DP、保存再会計、回帰、rankingは実行していない。

## 再利用できる成果

対象はlinear H6/H8、1 Å、STO-3G、DF rank19/28、T=.8。
H6のprefix5/10/15/full19、H8のprefix7/14/21/full28、q=1/2/4/8、既存S2/full S4・PR登録集合を対象とする。

|V2計画の項目|指定commitの保存成果|判断|
|---|---|---|
|H6の7 root記述|既存archive `ad52f9f6…` の7 rootを再利用|再生成不要|
|timed primitive receipt|H6 504、H8 708、計1,212個の値digest|既存表の来歴に再利用。H6の旧Phase Aを再実行しない|
|H6 104候補×A/B×両軸|416行、旧V2候補ID集合と完全一致|基本理論stageは再計算不要|
|H8|追加416行、7 preparationの旧hash一致|既存成果として参照。V2の新H8実行承認には読み替えない|
|finite-RTE law|H6/H8各72、計144 receipt、K2/K4全支持次数|保存lawを再利用。未観測次数を0にしない|
|保存shot・compiled費用への接続|10,480条件・metric行、うち5,736接続、4,744は保存shot不適格|旧shot再導出不要。欠測の0埋めをしない|
|counter/producer|旧V2とcounter bytes一致、H6/H8 provider共通6 source一致receiptあり|固定モデルとして再利用。provider全体同一とは扱わない|

exact Fractionを保持した理論JSONと保存接続gzipを正本として扱う。CSVは閲覧用decimal。
gzipの圧縮・展開後SHA256は元receiptと一致した。22個の保存artifactのbytes/SHAを記録した。

## 不足部分と、必要になる作業の種類

|不足・制約|現状|追加作業の必要性|
|---|---|---|
|native混合政策|表は均一A/B。実nativeのPF prefix=A・RTE tail=Bとは別|literal nativeの独立表が必要なら追加counter評価。今回の再利用・報告には不要|
|較正・PR/PF残差差の.05判定|`compiled=a+b*theory`、共通stratum残差、PR/PF比・rankingは未実施|互換な来歴を確定した後の保存値解析。新compile/samplingを必須とはしない。実施は別途判断|
|完全な数値wrapper同一性|全10,480行が `CONDITIONALLY_ALIGNED_METADATA_NOT_FULL_NUMERIC_ENVELOPE`|既存SOURCE/seed/event/phase/profile/variant receiptの静的整理が先。欠けた値を一致扱いしない|
|希少次数の個別費用寄与|全支持次数は期待値へ含むが、次数別の寄与表・compiled母平均保証はない|寄与表が必要なら保存lawに条件付けた追加解析。標本平均を母平均保証に置換しない|
|non-Clifford回転角分類|lowered RZ角の分類表なし。primitive receiptは値digestであり角配列ではない|旧/private角配列の有無を先に確認。必要なら限定した角の再構成への別承認。T/Toffoli数は対象外|
|T=.4/1.6のdeterministic費用|832行はT=.8のみ|旧T panelの保存receiptを静的照合可能。T=.8 B0Iで無条件代用しない|
|V1追加prefix候補|832行は各系の旧104候補|新V1候補の費用表ではない。V1のsignal計画とは分離|
|primitiveの完全payload|公開receiptはroot/block/time→digest|旧V2 adapterの完全payload validatorをhashだけで合格にしない。既存理論表の利用にpayload再生成は不要|

指定評価はgrouped DPを用い最大14,616 grouped transitions、上限1,000,000。
旧V2のcomponentごとのDPと遷移会計の定義が異なるため、上限の数字だけで同じ実行契約と認定しない。
binary64 lawのFraction正規化は数学的な真の確率の誤差保証ではない。
一般identity位相の0〜1 P/closed occurrence幅、経験的数値allowance、H6旧basis variant、
旧profile `4e32bb9b…` とcurrent `622ce4a8…` の差を保持する。

理論費用は `explicit_rz_sx_x_cx_v1` の固定分解であり、optimization1 compiler出力の同値予言ではない。
state preparation・measurement・routing・FT回転合成は除外されている。depth/T/Toffoli数は未算出。

## STOP・保全

既存評価のterminalは `COMPLETE_MANDATORY_STOP`、native exit0・child reap済み。
今回の静的process確認ではV1/V2と指定評価の実行scriptに一致する処理0。こちらの科学計算も起動していないため、停止信号の送信は不要だった。
既存結果・SOURCE・grant・ledger・H5・共有venvを変更していない。旧V2の未承認draftを実行可能化しない。

**V2の照合を報告してmandatory STOP。追加primitive生成・理論評価・回帰・費用再計算・次Phaseへの自動移行は行わない。**
追加解析の優先順位と範囲は、この報告後の利用者判断とする。
主担当の技術照合であり、別担当による独立reviewではない。subagents0。
