# A1 N3並行設計：判定基準と期待RZ費用保証

2026-11-06 JST。**静的調査・数式整理・設計文書・人工検査のみ。N3結果未閲覧。公開照合後mandatory STOP。**

## 成果物

- [N3_RESULT_DECISION_FRAMEWORK.md](N3_RESULT_DECISION_FRAMEWORK.md)：同target/ε/αの比較gate、A1対B1-OL、全実現RZ、有利・不利・未完了・判定不能と1campaignのclaim範囲。
- [EXPECTED_RZ_COST_GUARANTEE_PLAN.md](EXPECTED_RZ_COST_GUARANTEE_PLAN.md)：既存価格provider/会計の再利用、独立価格較正、全support上界、cal依存N、条件付き/無条件/STOP込み期待値、同時区間、最小追加scope。
- [正式N3契約の保存コピー](references/N3_EXECUTION_PUBLISHED_03928ce5.md)、[SOURCE.json](SOURCE.json)、[人工テスト](test_artificial_guarantees.py)、[結果](TESTS.json)、[raw log](TESTS.txt)。

## 固定した根拠と独立性

設計baseは `a167effd4a1192d9806ea0bde8e0f7af71a1ef55`。そのN3草案を採択済みと誤認せず、
ユーザー指定の正式[03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25/EXECUTION.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25/docs/research/o2_a1_n3_science_20261106/EXECUTION.md)
（Git blob `251eedffa2d55785adafcfec6e730fa19d230a34`）と条件を照合した。
PROJECT_MAPと研究概要を全文確認し、研究目的、A1のN1/N2/engine/N3接続資料、validation status/manifest、既存SOURCEを優先した。
正式共通指示はbaseに未収録だがユーザーworkspaceの指定原本を読み、SHA256をSOURCEへ記録した。
IDEの独立科学レビュー本文や未完成N3結果は本判定の入力にしない。

対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/N2保証ground、
L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、actual Q/selected policy、理想gate/wordモデル、
Qiskit1.3.0/(rz,sx,x,cx)/opt2/seed17/backend等なしのfull no-prep RZに限定する。
新しい数値的な物理結果を引用・再集計していない。

専用branch `codex/a1-n3-decision-expected-rz-20261106`、worktree
`/tmp/a1-n3-decision-expected-rz-20261106`、outputは本directory、cacheは
`/tmp/a1-n3-decision-expected-rz-cache-20261106`。
固定commitのsparse checkoutと公開済み正式契約だけを参照し、N3出力/cache/marker/lock/STOP/grant/ledger/実行processを開いたり変更したりしていない。
共有venv/cache/環境・他者processは変更せず、stdlibの軽い人工検査に恣意的なCPU/RAM/worker上限を加えていない。
Gitの専用branch/worktree管理と公開に必要な共有object/ref追加だけを行い、元worktreeのcheckout/indexを操作しない。

## 確定できた判定方針

正式campaign主比較は、全4方式両軸の返却保証と全価格が揃った場合のA1対強いB1-OLの両軸全RZ差とする。
train/pilot/cal/prodを全て含める。差の符号は当該campaignの実現有利/不利であり、期待改善の保証ではない。
shot cap超過、rank/range guard、途中STOPは未完了。保証前提破損や価格欠測は判定不能の理由を併記する。
A1を先に学習して最初の失敗で全体STOPする契約なので、未取得baselineを勝者や0費用方式にしない。
返却精度の成立、完走する確率、費用削減の成立は別のgateにする。

判定規則は結果を知らず固定したもので、N3開始前/完了前の登録とは断定しない。
今後結果を受け取っても本版を上書きせず、固定commitを引用して別報告を作る。
今回の設計方針の保存は、実行中N3の科学条件変更や新しい研究条件の採択を意味しない。

## 期待費用保証の実現方法と限界

既存SavedPrices、full builder/compiler/cache、CV凍結/cal/配分、TrialCost会計を再利用する。
全5層×X/Yの価格上界の根拠とfresh price-only較正があれば、同時価格CIと凍結計画の条件付き期待費用を構成できる。
手法全体の期待費用には、学習からの独立pre履歴反復とouter CI、全失敗history、production STOP補正が必要。
production測定を行わずquota×真の価格meanへ条件付き平均化し、必要なcompileはprice poolとpre pathへ限定する案を示した。
理想量子測定への期待費用移送は、engineの座標誤差を履歴couplingの費用幅に変換する条件を別に示した。

全support compiled RZ上界、実際のCI幅/必要n/J、N分布、返却率、STOP coupling、engine移送幅と実費は未確定。
観測最大や未compile sizeを上界にせず、条件付き期待費用を無条件期待費用と呼ばない。
安い中断を同じ成功率の資源削減と扱わず、価格/outer/abortの失敗確率を既存α_diagから流用しない。
数値的な期待費用・改善率、最終RPE総費用、CVの科学的優位性は認定していない。

## 検査と変更範囲

人工10件は、有限toy lawの全分散・層別CV相殺、adaptive Nと価格の依存、全段階/重複価格課金、
全support根拠の拒否、Hoeffdingの有限全標本検査、共有価格の差区間、失敗history保持、STOP補正、
engine bit誤差と不連続費用policyのcoupling、科学依存非importを確認する。
実RNG・実分子入力・real-state/Born・実full build/compile・保存数値再解析は0。
既存の純粋会計/価格radius helperを呼び、科学module/runner/元testは変更していない。
これは設計の人工検査であり、実科学validation、CI実行、独立外部再現ではない。

文書索引・研究概要・当日ノート・validation status/manifestに**設計資料として**追記する。
科学library/runner/artifactの新しい検証経路は追加しないため、既存pipelineの実行許可やstatusを変更しない。
専用branchでcommitしたが、通常HTTPS pushは端末の認証情報不足（could not read Username）で失敗した。
接続済みGitHubのGit Data APIで同じ親と同じtreeを公開する代替手段を使用し、remote ref SHAと変更した全blob/本文を照合する。
API側のauthor/committer metadataでcommit SHAが変わる場合も、元の親・全treeの一致を確認する。
最終remote SHAと公開照合結果は最終応答および専用cacheのPUBLICATION_VERIFICATION.jsonへ記録する（commit自身のSHAを自己埋込みしない）。

## 次に必要な承認とSTOP

次は、期待値のestimand/返却率/信頼予算を選び、価格上界の取得方針と最小price-only scopeを採択して明示許可すること。
必要な場合だけ独立pre履歴反復・word-only STOP評価を追加scopeとする。新しいgrant/承認証明発行は要求しない。
この提案は実行許可ではなく、詳細科学レビューも開始しない。

**公開照合後はmandatory STOP。N3介入、新科学計算、追加campaign、条件採択、詳細科学レビューを開始しない。**
