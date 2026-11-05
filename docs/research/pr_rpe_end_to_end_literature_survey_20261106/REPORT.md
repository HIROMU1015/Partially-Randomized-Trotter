# PR＋RPE総費用削減に向けた文献差分調査：報告

調査日：2026-11-06 JST。基点は [3571477093441ac4852166fe829f8ccc5db5ec4f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3571477093441ac4852166fe829f8ccc5db5ec4f)。専用branch：`codex/pr-rpe-end-to-end-literature-20261106`。本調査を含む固定公開commitと取得照合の結果は最終チャットに記す。

## 成果物

| 文書 | 内容 |
|---|---|
| [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) | PR/UWC/SPRINT/IS、PAI/TETRIS、SE-QPE/QSP-PE等の対象・方法・保証・資源scopeと本研究への接続 |
| [PRIOR_ART_COMPARISON.md](PRIOR_ART_COMPARISON.md) | 強baselineと同じ改善機会、既知対応、保証の差、負結果・STOP |
| [END_TO_END_COST_ANALYSIS_MAP.md](END_TO_END_COST_ANALYSIS_MAP.md) | 単回価格・bias・normalization・必要shot・全roundの依存関係。概念図と既存式の再掲 |
| [RESEARCH_OPPORTUNITIES.md](RESEARCH_OPPORTUNITIES.md) | 6方向の条件付きの問い。機構・全費用・重複・反証・資産・追加作業を整理。順序は優先順位ではない |
| [SOURCES.md](SOURCES.md) | 23文献の著者・版・出版情報・URL・確認節・アクセス制限・再利用/追加区分、固定内部証拠 |
| 本REPORT | 差分、到達点、限界、保全、公開と停止点 |

## 1. 目的と調査範囲

元の物理Hamiltonian Hの厳密基底状態を利用できる仮定を候補とbaselineで共有し、状態準備を除くRPE全shot/full-circuitの期待費用を目的とした。未知のE₀やgapを無料の入力情報としない。異なる資源指標、固定時間G、query数、RMSE、energy成功確率を区別した。精度・失敗確率・分子・compiler・resource metricの新しい固定値は決めていない。

新文献探索前に、PROJECT_MAP、適用AGENTS、研究概要全文、研究目的・証拠status/manifest、正式共通指示2026-10-25の指定節を確認した。指定統合5成果物、旧70研究/Phase 2A・2C/method-delta、11/1科学landscape/gap、統一索引、Track Bの関連比較を再利用した。

統合CSVの129記録/48列と関連bindingを読取り、143資産群/179版の既存再利用地図を参照した。これらは独立実験件数ではなく、今回全資産を再実行・全原証拠を精読したという意味でもない。SOURCEの確認は対象接口の静的読取に限定した。文献は23件を同じ深度で全文読了したものではなく、本文部分確認・要約・過去照合の継承を個別に表示した。

## 2. 今回追加確認した重要事項

1. **全費用の結合は既知**。PR原著App.Eはround合計とnormalization/shot・合成残差の交換を扱う。UWCは表現＋split＋RPE費用を目的にし、round別表現最適化と古典負担にも明示的に触れる。「jointに最適化する」「round別に変える」だけでは方法差にならない。
2. **UWCの出版状況が更新**。2026-10-28のPRX Quantum受理ページを確認した。arXiv v2と受理原稿の全本文差分は未照合であり、外部論文の更新は来歴未結合のrepository旧UWC値を復活させない。
3. **SPRINTの減衰/合成解析を本文で確認**。App.Fは位相shift・spectral weight・減衰を区別する。XASにおける良い資源推定を、そのまま元HのRPE成功保証にできない。
4. **ISの目的によってlawが違う**。resource-optimal ISのp/√cはworst-case二次momentのnet cost、cost-aware qDRIFTのh/Cは異なるsimulation/experiment会計を扱う。両者の保証範囲を分け、有限confidenceと全path取得費を未解決点として残した。
5. **連続時間PAIのHadamard部分はTETRIS**。2026-10-23のv3はAlgorithm 1訂正を告知し、本文/App.CではTE-PAIによる準備とoperator平均を使うTETRISによるenergy測定を分ける。今回の状態準備除外では準備改善を成果に数えない。
6. **controlとsignal側の近接対照を追加**。SE-QPEは共通固有基底・真空reference、QSP-PEはcontrolled unitary/query条件を持つ。本PRのrandom path/finite meanへ保証を移せるかは未確認。SEの追加register/CSWAP、QSPの多重query/合成費を含む必要がある。
7. **RPE原著訂正を明示**。旧分枝失敗解析のπ/(2k_j)条件に訂正がある。既存decoderが不適合だという新判定はせず、将来の保証照合で使う原資料として記録した。

一次資料の具体的位置はSOURCESのclaim/locator表へ追跡できる。上記は文献に関する追加情報であり、本repositoryにおける新しい科学結果ではない。

## 3. 既存結果との照合・研究機会

既存のP-A interval、P-C/P-D/R3、反可換pair、sampling freedom、singleton、FR、C1等の限定負結果・既知対応・STOPを引き継いだ。F2の単回資源とN未資格、研究Cのnominal/own-DF state/固定word、RPEの条件付き全round成功・energy接続と最終費用未決定を区別した。条件付き18roundの後続成果を未着手に戻していない。

| 整理した問い | 追加の知見になり得る性質／未解決点 |
|---|---|
| O1 表現・tail・phase・native price | UWC等の強対照後にも残る有利条件。元H state/fit誤差と取得費 |
| O2 samplingアクセスと有限confidence | 非加法whole-path価格を安く利用する性質。N/weight・生成費の接続 |
| O3 bias・attenuation・合成 | 既知processing/PAI後の成立条件。長roundでのbackbone/shot交換 |
| O4 異round構成のenergy整合 | 同じE₀/branchを取得可能な情報で結ぶ性質。固定構成＋配分対照との差 |
| O5 reference/controlとrandom path | SE-QPEの条件をPR平均で満たす範囲。追加qubit/CSWAP/routing |
| O6 代替signal設計の比較境界 | ideal-query保証から近似oracle/full priceへの接続。RPE以外を扱うかは未採択 |

既知法の実装追加だけで済む作業と、追加のalgorithmic性質が必要な研究上の問いを分けた。O1〜O6はいずれも正式な新方式・科学的優先候補ではない。新規性・GO/NO-GO・論文化可能性を最終判断していない。

## 4. 証拠不足と確認の限界

- 外部文献の保証をこのprojectへ自動移送していない。元H stateとown-DF ground、channelとcoherent mean、rawとscalar補正平均、model価格と全law期待価格が主な接続境界である。
- SE/QSP/phase2等の指定旧調査集合への追加は、repository全体で参照が一度もなかったという主張ではない。Google Scholarの全被引用集合や世界の全関連論文を網羅していない。
- 出版社本文の一部はredirect/取得不可だった。中心claimは取得できたarXiv本文または取得範囲を明示した既存固定照合へ帰属した。publisher最終版とarXivの全行差分は未確認。
- 旧SOURCE/raw/grant/STOP、全private host原本、実験独立性を新たに全面監査していない。旧C_useをrigorous boundにせず、H12係数・最終総費用優位を主張しない。

## 5. 文書検査・保全・公開

文書検査には旧Phase 2A `verify_documents.py` の相対link・秘密文字列・Git blob照合の処理方針を再利用した。同ファイルの標準ライブラリのみの`git`/`blob_oid`を使用し、固定namespace/JSON schemaに依存する旧validator全体や科学runnerを起動していない。この6文書に適合するlink/ID検査だけを一時的に追加し、新しい管理システム・承認証明物・恒久validatorは作っていない。

公開対象はこの新directoryの6 Markdownのみ。元の作業領域の未commit変更を保持し、指定commitから共有objectを読む隔離checkoutで専用branchを作成した。共有object方式は独立backupではない。元科学branch、main、統合5成果物、SOURCE/input/raw/result/grant/ledger/marker/STOPは変更しない。新たな科学結果・正式方針・evidence statusの変更がないため、研究概要・manifest・既存索引は更新しない。本REPORTを新調査の入口にする。

6ファイルのUTF-8/末尾改行、相対link/内部見出し、L01–L23の登録、文書に明記した固定Git参照、非公開path/認証情報の混入を検査し、問題を認めなかった。最終版では96 Markdown linkを含む。これは文書の技術検査であり、文献の全URLに対する再downloadや科学的妥当性の再審査ではない。差分はこの6文書への追加だけを許容する。

公開はHIROMU1015/Partially-Randomized-Trotterの専用branchへの通常pushに限定する。公開後、remote refのSHAと6ファイルのGitHub取得内容/bytes/blob一致を確認して終了する。自己commit SHAを本文へ埋め込む再commitはしない。

## 6. 未実施・mandatory STOP・次の担当

新しい科学計算、保存値再解析・改善率/総費用計算、Hamiltonian/state/reference生成、simulation/sampling/RNG replay、回路build/compile/native synthesis、旧runの再実行・retry/resume、algorithm正式設計/実装、非自明な新証明、GPT/Claude詳細科学レビュー、新規性・研究方向の正式採択は実施していない。

**mandatory STOP：文献差分調査・既存研究照合・研究機会整理・6成果物公開・取得一致確認が完了した時点で停止する。**

次の担当はユーザー／GPT。必要な問い、応用実装か追加の科学的性質か、RPE以外の推定器を比較するか、将来の対象・精度・失敗確率・資源指標を選ぶ。詳細レビューと新科学計算はそれぞれ別の許可が必要である。Codexが自動開始する次研究作業はない。この調査の完成のために新しい科学計算や詳細レビューは必要ない。
