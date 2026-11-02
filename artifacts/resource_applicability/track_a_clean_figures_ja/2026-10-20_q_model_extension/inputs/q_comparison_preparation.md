# PFのq依存モデルと直接実測：比較準備

2026-10-20。基点`2619688b26c1146ae8c32d31d272282bfd1de407`。
保存788軸を登録済みH6/H8の8モデル定義へ照合し、保存予測と直接実測を結ぶCSV/JSONを作成した。新規モデル予測・compile・fit・誤差/比率解析・費用再会計・ランキングは未実施。詳細科学レビューと最終evidence freezeも開始していない。

## 対応表と分類

[packet](../../artifacts/resource_applicability/track_a_pf_q_model_comparison_preparation/2026-10-20/README.md)の`axis_correspondence.csv/json`が788行の主表。分子/DF系列、PF公式、prefix、q、cosine/sine、control、compiler/profile、入力hash、回路fingerprint、実測SOURCE、worker結果SHA、保存予測文字列、モデル式/係数、旧利用点のrole、比較条件を記録する。複数モデルや予測の来歴を保持できるよう、モデル定義と保存予測レコードを別表にした。既存推定値を実測値で上書きしていない。

| 系列 | 保存実測 | 比較可能 | 条件付き比較 | 現状比較不能 | 内訳 |
| --- | ---: | ---: | ---: | ---: | --- |
| H2 DF3 | 72 | 0 | 0 | 72 | 同一系列の登録qモデルなし |
| H4 DF10 | 122 | 0 | 0 | 122 | 同上 |
| H6 DF19 | 214 | 0 | 192 | 22 | モデル定義はあるが22軸の予測未保存 |
| H8 DF28 | 314 | 0 | 258 | 56 | モデル定義はあるが56軸の予測未保存 |
| H4 legacy DF12 | 66 | 0 | 0 | 66 | DF10と別系列、同一系列の登録qモデルなし |
| 合計 | 788 | 0 | 450 | 338 | 予測未保存78＋同一系列の登録モデルなし260 |

「比較可能」は、保存予測が存在し、照合対象のidentityに未解決条件がない点と定義した。450軸は保存値同士の数値比較を将来行える対応であるが、モデルanchorから現在のruntimeへのbinary同一性を完全には証明していないため、無条件の比較可能点にしない。「比較不能」は今回の保存資料のみでは比較用の二つの値が揃わないという意味で、将来の取得不能を意味しない。

対象はlinear 1 Å/STO-3G、tol-only DF3/10/19/28と別系列legacy DF12、T=0.8、P=0、symmetric_directional。指標は**状態準備を除いたfull Hadamard wrapperのcompiled RZ数**。shot重み付けした総費用ではない。PFの決定論的prefixとqを保持し、PR sampling/RTE費用や旧shot会計へ接続していない。

H6の優先確認点`B0_p11_q4_2nd`はcosine/sineの両軸で保存予測と新実測を対応付けた。εに応じた費用順位は再評価していない。同じ候補の直接ゲート数と、ε別の費用比較を区別する。

## 固定モデルと既存利用点

`model_registry.json`は元の8モデルを転記したもの。S2は`a+b*q+d*prefix*q`、S4は保存モデルの`a+3*b*q+3*d*prefix*q`。係数を今回評価していない。

| 系列・method | (a,b,d)の保存文字列 | 軸 |
| --- | --- | --- |
| H6 B0 | (797,109,9708/5) | cosine/sine |
| H6 B0I | (797,269,9613/5) | cosine/sine |
| H8 B0/B0I | (1592,1314,25271/7) | cosine/sine |

H6モデルのS2適用prefix6–15、S4適用prefix11–15、q=1,2,4,8という宣言を保持する。保存予測のある192軸は宣言範囲内。予測未保存22軸のうち14軸は宣言範囲外であり、将来の新規予測では外挿方針も承認する必要がある。H8は凍結ファイルに完全な適用domainの宣言がないため、保存予測点としてのみ対応付け、新しい適用範囲を推定しない。S4の計178対応軸（H6 74/H8 104）はschedule composition仮説を継承する。

`model_used_points.csv/json`にはH6 TRAIN_ANCHOR 28、DECLARED_VALIDATION 4、H8 TRAIN 22、HOLDOUT 2、追加のPRIOR_DIRECT_VALIDATION 18をrole別の証拠行として記録した。roleの重複を許し、件数を独立な標本数とは解釈しない。これらの候補/軸IDと新788軸との直接重複は0。このID照合だけで科学的な独立性や選択効果の不存在を保証しない。

H6の宣言モデルprofileはB0=`ab3c5301…`、B0I=`622ce4a8…`で、今回H6直接実測のprofile hashとは異なる。H8の名目profile hashは一致するが、モデルanchorの全runtime binary同一性は未証明。compiler hashは対応450軸で同一。設定hashの一致と実行環境全体の同一性を区別して保存した。

H6保存予測の`model_sha256`はモデルファイルbytesのSHA256、H8では元JSONの数値型を保つ正規化JSONのSHA256だった。H8のfile SHA=`a2f0ba10…`とcanonical SHA=`8420b9f2…`を別fieldに保存し、既存`h8_phase_b2_contract.digest`の定義を出典に照合した。単なるhash表現の違いをモデル不一致と扱わない。

## 旧実行の制約

| 実測SOURCE | 保存受入軸 | 今回の扱い |
| --- | ---: | --- |
| `220b9644df6e114116737fd041973973fcf86cf5` | 572 | task受入とworker exit証拠あり。run全体の元terminal/最終gate欠測を継承 |
| `0c5fcc83ef55045c2c64746aadfaf3128667b058` | 110 | 独立した終了gate PASS |
| `9c65fb312891d8334037f30ef953336c287bc262` | 106 | 独立した終了gate PASS。うちH2 20軸に旧signal未整合flag |

条件付き450軸のうち428軸は初回SOURCE、22軸はretry216 SOURCE。初回572軸を後続の最終gateで全体PASSへ読み替えない。実測の各行は保存workerの受入ゲート数・fingerprint・compiler・task IDに一致することを確認し、worker結果SHAを保持した。新しいgateや科学的な再認定は行っていない。

H2の20軸はB0/B0I、prefix0、S4、q1/2/4/8/16の両軸。正式Tのnative回路を保存exact_truncatedで確認した取得値であり、旧merged classical signalの不整合は未解消。`requires_saved_classical_signal_reconciliation=True`を残し、旧signalに基づく総費用・適格性・shot・ランキングへ採用しない。この20軸は今回は同一系列モデルがない260軸にも含まれる。signal修正の承認をモデル比較承認へ混ぜない。

旧profile等の照合待ち48軸は`excluded_48_axes.csv`へ転記し、788軸との非重複を確認した。再取得もprofile一致の新認定もしていない。旧SOURCE/raw/grant/ledger/marker/STOP、既存モデル、原図、論文主表を保全した。

## GitHub公開確認と公開情報検査

GitHub read-only APIと`git ls-remote`で、repositoryがpublicであり、基点2619688bと承認済みbranch`codex/pf-direct-compile-retry106-20261020`のheadが一致することを確認した。観測時刻と使用手段は`github_publication_receipt.json`に保存した。

本会話の「この80ファイルの公開を承認」と基点の`publication_scope_final.json`を照合し、直前承認commit a80f89b4以後の累積80パスが承認範囲と完全一致することを確認した。`sealed_contract_and_grant.json.gz`2件、worker/ledger証拠等を含む全80ファイルを固定Git bytesから読み、gzip展開後も秘密鍵・token・credential URL等のpatternとJSON認証fieldを検査した。該当patternや非空認証fieldの検出は0、範囲外公開の検出も0。旧圧縮ファイルは変更していない。

実行hostの絶対path、runtimeファイルhash、科学実行の`runtime_authorization/approved_by_user`は含まれる。これらは承認された公開80パス内の実行metadataであり、GitHub用認証情報とは区別した。検査範囲はこの80ファイルと基点/承認branchの公開確認であり、全Git履歴・すべての秘密形式の不存在を保証しない。詳細は`publication_review.json`。削除・履歴変更・force-push・新規公開はいずれも0。

## 構造検証と次段階

新12テストで、788軸の一意性/394完全pair、系列非混合、CSV/JSON/元実測文字列一致、450保存予測と元出典の文字列一致、typed model SHA、旧572/20/48のflag、学習/検証点role、人工圧縮credential検出を確認した。runnerの`--check`は固定Git証拠からの再転記と一致を確認するだけで、科学APIを呼ばない。検証receiptとpacket hash一覧は同packetに保存する。primary agent自身の構造検査であり、独立した科学レビューではない。

`next_numeric_analysis_scope_DRAFT.json`は未承認の範囲案で、有効grantや起動commandを持たない。次担当ユーザーが、まず**条件付き450軸の保存値を用いる誤差/相対誤差・系列別要約・描画**を認めるか判断する。元run欠測、profile/runtime差、S4仮説を保持し、解析条件・数式・SOURCE/input SHAを実行前に固定する必要がある。

予測未保存78軸は新規モデル評価の別承認が必要で、H6のdomain外14軸とH8のdomain未宣言を先に扱う。同一系列モデルのない260軸へのモデル選択/新fit、旧H2 signal修正、旧48軸照合/再取得はそれぞれ別作業。今回の比較準備から自動実行しない。

この新packetと資料の公開は未承認で、専用branchへのlocal commitのみ。次担当Codexは、新たな明示承認後に、その承認範囲だけを準備・実行する。現在は**mandatory STOP**。
