# A2 実用性preflight：ユーザー／GPT向け判断用ハンドオフ

2026-10-17依頼。[報告書](practical_preflight_report.md)／[認証情報・SOURCE対応表](certificate_source_matrix.md)／[固定証拠索引](evidence_index.json)／[公開receipt](publication_receipt.json)。**今回の技術分類は `INPUT_CLASS_FOUND_CERTIFICATES_INCOMPLETE`。** 既存の記号的step優位条件を実分子で評価したものではなく、研究採択・新しい科学レビューでもない。

## 1. 判断に必要な現在地

| 問い | 回答と限界 |
|---|---|
| 有効な入力classはあるか | 固定N、またはfragment別保存則を確認できるNα/NβのHermitian一体＋signed DF平方。K0の外側scalar shiftを用いる既知経路があり、論理Gaussian/対角/controlled部品もある。具体的な認証付き分子入力はまだない |
| 取得済みなのは何か | RF0/RF1/EW2の固定紙上監査、同一入力の強化Cbar2対照への記号条件、PSD化・行和の既知構成法、実SOURCEと保存DF schema/ファイルidentity。実測費・認証値は取得していない |
| 何が足りないか | 元H/DFのtarget選択と誤差会計、sector/scalar/normal-orderの同値証拠、L/U/f_j/Λ、同分割K_SOS/Cbar2の生成と丸め証拠、state access、固定S2 QPE/readoutと総費用 |
| 次の数値照合で判断は変わり得るか | 同じ証拠水準で両側を評価できれば、shiftの保守性・W/Λ・K_SOS/Cbar2の価格と認証step条件が実用上の障害かを判断する材料になる。現時点では良好な値や改善を予測できない |
| 今すぐ実行できるか | できない。認証情報の接続手順が未完成。今回SOURCE・実験契約を作っておらず、A2も未採択、科学実行未承認 |

## 2. 解決できた技術的な誤認と、未完の作業

SOURCE実読で、DF平方は符号付きであり自動PSDではないこと、sector indicesは保存則/量子encoderの証明ではないことを切り分けた。DFの4番目の戻り値をconstantへ加えない既存処理も確認した。回路用DFModelにはfull値をcorrection名のfieldへ入れるbridgeと、correction-onlyの生成経路があるため、その混用は避ける必要がある。既存callerの誤りを認定したものではない。

数値Hermitization・eigh・norm helper・ground cache・file hashはそれぞれ有用だが、元Hの認証ledgerへ自動的につながらない。N3のexact witness checkerやsynthetic saved oracleを流用して「分子用certificate完成」とはしない。新しいSOS/BLISS最適化や回路生成規則を発見したという結論もない。

残るadapter・validated arithmetic・certificate検証実装は技術作業になり得る。ただし、今回の「科学SOURCE実装禁止・新certificate数値処理禁止」を超えるため、必要な作業を特定するところで止めた。既存技術問題を理由に無制限な候補探索や閉じた研究の再開へ進んでいない。

## 3. 科学的な選択が必要な点

1. **保証対象**：元電子Hamiltonianか、固定binary64 DFモデルか。前者ならDF・入力/対称化・回路モデル誤差を別途認証する。後者の比較成果を元分子の厳密保証へ格上げしない。
2. **証拠水準とaccess**：厳密/検証付きの境界が必要か、明示した仮定下のengineering調査までを一段階とするか。EW2の厳密ground支持を、保存数値stateで置換してよいとは今回証明していない。state準備は無料入力にしない。
3. **研究としての比較軸**：固定回路の認証step改善をまず問うか、取得費・状態準備・native・readout込みでのみ価値を認めるか。同じstrong baselineと情報価格を適用する。
4. **不足証拠を準備する価値**：既知DF shiftに留まるclassで前処理/certificateを整える意義があるか。classや既知構成の存在だけで新規性・実用優位を採択しない。

これらはユーザーの未採択の選択であり、本preflightは回答を代行しない。GPTによる重要科学的判断が必要とユーザーが判断する場合は、開始の別途明示承認を受ける。同じRF/EW2証明のレビューを自動で繰り返す必要はない。今回GPT/Claudeを新たに起動していない。

## 4. 後続を依頼する場合の最小の順序

これは作業候補の説明であり、実行契約・承認・新しいthreshold/call予算の設定ではない。

| 段階 | 担当・必要な判断 | 取得すべき証拠／STOPの理由 |
|---|---|---|
| 対象と証拠水準の選択 | ユーザー。必要なら別承認した科学判断 | 上記target/state/access/比較範囲。A2採択や数値実行を兼ねない |
| 接続手順の準備 | 別途指示後のCodex | raw→DF→scalar/sector→PSD ledger→L/U/f_j→同順序K_SOS/Cbar2の再現可能な仕様。新SOURCE/契約を準備するならその範囲を別途承認。欠落や同値性未解決なら実行に進まない |
| 最小の科学的照合の設計 | ユーザーが対象条件・精度・比較範囲を採択 | 一つの共通入力・分割で認証と両側stepの比較ができる設計。分子/geometry/rank/閾値等は今回選ばない |
| 証拠取得と限定比較 | 別途の科学実行承認後のCodex | 認証生成/検証費とL/U/f_j/Λ/K_SOS/Cbar2、モデル・sector同値性、両側の同情報条件。truthや結果一致を使った閾値調整・unapproved retryをしない |
| 資源判断 | 結果後のユーザー。必要なら別承認した科学判断 | 認証step差と実PF bias差を区別する。state/native/powers/readout価格がなければ総資源優位は未解決のまま |

最小比較で得られるのは、固定classの証拠取得と保証選択が有望かという限定的な判断材料である。仮にstep条件を満たしてもbest-known envelope全体や総費用への優位を認定しない。条件が不成立でも、より良い入力/保証の数学的不可能性やencoding一般の終了とはしない。

将来の新契約でも共通指示§8.4に従い、恣意的なCPU/RAM/worker/wall上限を既定で追加しない。実際の空き資源・演算構造・精度の再現性を踏まえて高速化し、他job/共有環境へ干渉しない。既存契約の制限をこの方針で遡及変更しない。科学call・scope・保存・STOPの境界は別に必要である。

## 5. 現在の状態・保全・終了

正式状態の固定正本は[decision `67576526…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)。A1 CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行を維持する。Exact Encodingの限定構成のREDUCES_TO_KNOWN_METHOD、他のCLOSED/STOP/PAUSED/HOLD、①②③の状態も変更しない。

専用branch `qc-a2-practical-preflight-20261017`、独立作業領域 `/tmp/prt-qc-a2-practical-preflight-20261017`、基点 `db1819587df36ee8833b37227ca0083faa0408ec`。変更はこの新規専用ディレクトリのみ。SOURCE・raw・科学契約・authorization・grant・ledger・markerと原本を全て保全する。研究概要・PROJECT_MAP・validation manifestを変更しない。

入力指示は[改行を含むbyte保存copy](inputs/user_instruction_a2_practical_preflight_20261017.txt)。最新共通運用指示2026-10-17は前回監査の固定copyとローカル原本のSHA一致を確認した。必須指定資料の欠落はない。未取得なのは実入力の数学的certificateと費用であり、文書の未取得と区別する。

文書本文・証拠索引のpayload commitを公開し、remote HEADと公開bytes/SHA256を照合した後、receiptを子commitへ保存する。receipt自身のcommit/hashの自己参照は行わず、最終HEADの公開同一性は最後に再確認してチャットで報告する。親HEAD・local branches・worktree registry・index/exclude・tracked files/選択原本と、独立作業領域の全base blobを照合する。

新規科学計算・保存値再解析・新数学保証・科学SOURCE・科学実行契約・authorization・grant・marker・simulation/native compile・新重要科学レビュー・A2採択は**各0件**。

**次の担当はユーザー。** 不足する証拠の接続手順を別途準備するか、その前に上記の科学的選択を依頼するかを判断する。今回の完了はpreflight資料の公開まで。**Mandatory STOP。**
