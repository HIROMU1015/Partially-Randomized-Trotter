# q依存費用モデル：100 scopeの条件付き成果採用と証拠統合

2026-10-20。基点・採用結果は **`3983fe4f1ede003953735e757d2c098c12b4644f`**。
専用branchは `codex/pf-q-model-evidence-integration-20261020`。
ユーザー指示により100 scope（重複除外40条件）を条件付き研究成果として正式採用し、既存結果の文書・証拠統合だけを行った。新規科学計算・数値解析・再ランキング・再描画は0。

## 統合内容と科学的主張

[共通プロトコルv2補足](track_a_protocol_v2_pf_cost_supplement_20261020.md)を旧70から100 scopeへ更新した。
正式v2本文・科学条件・受入閾値、既存SOURCE/raw/grant/ledger/marker/STOP、論文主表・Figure2・科学図表はbyte不変である。
旧70版は基点commitのGit履歴に保存されている。旧科学receiptや結果を新しい値へ上書きしていない。

対象はlinear H6/H8、隣接1 Å、STO-3G、tol-only DF19/28、T=.8、P=0。split L_Dは登録prefix、PF q1/2/4/8、δ=.8/.4/.2/.1。
ε=.05/.01/.005/.001/.0005、signed/noncancellation bias・allowance倍率1/10、H6登録domain32/96/136、H8 domain32/184を維持した。
full cosine/sine Hadamard wrapperのcompiled RZ×保存analytic shotを使い、PRは決定論的prefix・確率的RTE tail・full wrapperを含むH6 union8/H8 union6各n32の保存mean/SE/cohortを保持する。

| 採用する保存点結果 | 全100 scope |
| --- | ---: |
| 最良PF候補変更 | 2/100 |
| q・PF次数変更 | 0/100・0/100 |
| 最良PF総費用変更 | 50/100 |
| 候補全体の順位変更 | 50/100 |
| PR/PF優劣逆転 | 0/100 |
| PR低費用 / PF低費用（両政策とも） | 90件 / 10件 |

最良候補変更2件はH8非相殺、ε=.0005、strengthened184、倍率1/10のB0I p16→p15、q4/S4保持。
この結果はモデル推定・既存実測の混在政策から保存直接単価へ置換した条件付き比較である。モデル固有誤差だけの検証、純モデル反実仮想、全順位不変、母平均優位の証明とは主張しない。
100は候補集合・精度・倍率・cohortが重複するscope数で、独立100実験の件数ではない。40 endpointも独立な40 sampling実験とは扱わない。

## SOURCE・数値来歴・SHA

| 区分 | 結果commit | 科学SOURCE | 採用方法 |
| --- | --- | --- | --- |
| 旧70 | `85433c48fd4c37d4cb4594391da422558ac3ce7f` | `19f2c5a9409db69fd8bd6500eb4228fefa00145c` | H6 signed30・H8両bias40の保存行を転記。今回も段階B時も費用・比・順位を再計算しない |
| Stage A | `c53dd16ae16413182c9d3080f82f80237231a248` | `85e250302d15b427731a50ddbe321fc280ec5f1a` | 旧B0I12候補の120 endpointを適格34・不適格86・未解決0へ確定した受入済み結果 |
| 新30 | `3983fe4f1ede003953735e757d2c098c12b4644f` | `c0d8ca8ecd3fc29528ca8424f8b8d953d06eff91` | H6非相殺30 scope。旧unknown120行だけへStage Aを接続し、既存712費用exact再現と新34費用の構成を区別 |

新34適格endpointの費用は今回段階Bで構成された新baseline来歴で、旧全domain最良値はnullだった。過去の全domain winnerを保存結果から再現したとは呼ばない。
旧Dのtruthinessで空になったunresolved配列を解決証拠とせず、Stage A/Bの新結果で解消した。停止SOURCE224fbba0の部分結果・旧STOPは保全した。

[新論文manifest](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/paper_manifest.json)は既存主表・Figure2・1576行のpaper data manifestを参照し、100 scopeのCSV/JSON・図6ファイル・入力/結果/監査をSHA256で結合する。
[CEM21行](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/claim_evidence_matrix.csv)は旧14行をそのまま保持し、7つのscoped claimを追記した。
[台帳v4・338行](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/evidence_ledger_v4.csv)は旧316行をbyte prefixとして保持し、22件の保存literal・JSON pointer・SOURCE・結果SHAを追記した。値の足し直し・比の計算・再順位付けは行っていない。
[現行overlay](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/claim_evidence_current_overlay.json)で、旧CE5のH6欠測記述を歴史的状態として限定更新した。他の未解決事項は自動昇格しない。

2図は既存の[PF費用変化](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PF_cost_change100.png)と[PR/PF比](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PR_PF_ratios100.png)を参照する。PNG/SVG/PDFの元bytesを変更・再生成せず、掲載用captionはv2補足で再利用した。
[GitHub照合記録](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/github_observation.json)と[公開監査](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/publication_audit58.json)に固定blobとローカルSHAの対応がある。

## GitHub公開状態・承認記録・公開内容の監査

GitHub公開repositoryと専用branch `codex/h6-noncancellation-stage-b-rerun-20261020` のheadが採用結果commitに一致することを確認した。
[固定commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3983fe4f1ede003953735e757d2c098c12b4644f)の58 pathすべてで、公開Git blob SHA・sizeと固定ローカルGit objectのbytesが一致した。
summary・completion audit・native exit evidenceはraw直接GETでも完全一致。新実行SOURCE/test全9 fileの公開blob、旧70/Stage AのSOURCE commitの公開存在も確認した。科学処理・原compiler outputの再実行ではない。
公開済みcommitは再pushしていない。

**公開は確認できたが、利用可能な記録だけでは58ファイルの明示的な公開承認receiptを確認できなかった。**
当該`publication_scope.json`、STOP等は実行当時の`new_publication_authorized=false`を保持し、科学実行指示receiptも公開承認とは分離されている。
このチャットではpushコマンドの要求・提示を確認できるが、別の明示承認・手動公開の主体や経緯までは確認していない。
したがって「ユーザーによる未承認公開」と断定せず、公開状態と承認確認状態を別欄に記録する。今回の正式な科学採用を、過去または今後の公開承認へ自動変換しない。

58ファイルのbytes/text、PNG text metadata、PDFの展開可能なFlate streamを対象に資格情報パターンを検査し、検出0。
一方、ユーザーhome・一時領域の内部絶対パスは次の5ファイルに88箇所検出した。該当値そのものを新しい報告・監査へ転載せず、repository相対path・位置・分類・literal hashだけを記録した。

| 当該58ファイル内のpath | 内部パスの検出箇所 |
| --- | ---: |
| VALIDATION_STATUS.md | 1 |
| artifacts/validation_manifest.json | 4 |
| 段階B再実行packet/artificial_tests.json | 1 |
| 段階B再実行packet/sealed_contract.json | 81 |
| scripts/run_h6_noncancellation_stage_b_rerun.py | 1 |

上記packetは`track_a_h6_noncancellation_stage_b_rerun/2026-10-20`。既存の索引・manifestに含まれる歴史的記録も検査対象に含む。
詳細は[位置・分類・hashの記録](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/publication_findings58.json)を参照する。
SOURCEや算術runtimeの証拠として保存されたパスと、使用可能な認証情報を区別する。パターン検査の0件を「機密情報が絶対にない」とは表現しない。font符号化PDF text・画像pixelのOCRまでは実施していない。
旧履歴の変更・削除・force-push、grant/ledger/STOPの編集、他チャット履歴やprivateアカウントへの広域調査は行っていない。
既存58ファイルの公開経緯と内部パスを公開したまま保持してよいかはユーザーの確認事項とする。修正が必要でも、今回無断で旧証拠や履歴を書き換えない。

## 構造検証・保全・限界

新しい7人工testと文書・構造・SHA検証はPASS。対象は保存literalの一致、旧70の転記、新30との区別、旧CEM/台帳のprefix保持、証拠path/JSON pointer、正式v2と主表/Figure2・旧SOURCE/結果/grant/ledger/marker/STOPの保全、公開blob identityと監査の区分である。
`python3 scripts/integrate_pf_q_model_evidence.py check`で再確認できる。科学runnerをimportせず、モデル予測・費用会計・比・fit・統計・ランキングを計算するAPIは呼ばない。
検証はprimary agent自身の作業であり、独立した別担当の科学レビュー合格や外部再現とは呼ばない。

H6 profile差、H8/model runtime同一性未証明、S4合成仮説、初回572軸の最終gate欠測、旧profile待ち48軸、経験的bias/allowance、固定cohort n32・登録集合の制約を保持する。
平均費用推定値とn trajectory取得合計費用、固定集合の点最良と全域最適・母平均保証を区別する。未観測希少event・全support分散・FT/energy/RPE総費用を保証しない。
正式v2科学条件は変更せず、詳細科学レビュー・最終evidence freezeは未開始。

## 次の担当

**ユーザー**：既存58ファイルの公開承認の経緯と内部パス開示の許容範囲、新しい文書統合成果物の公開範囲を確認する。
**Codex**：承認範囲が確認できた新規成果物のみをpushする。公開済みcommitは再pushせず、既存履歴の修正や追加科学計算を自動実行しない。
今回の研究資料統合を完成させるための追加科学実行は不要。

**mandatory STOP。**
