# Track B：Parent allocationの科学レビュー前・最終技術準備

## 目的

Parent allocationのsource準備が完了したので、次のGPT重要科学レビューに先立ち、**科学的判断を必要としない技術作業を可能な限りCodex側で完結**してください。

今回の目的は、比較計画を科学的に採択することではありません。

目的は次の3つです。

1. 既存native実装・cache・strict phase guardとの接続について、技術的に解決できる問題を解決する。
2. 現在の最小比較計画案の技術的な弱点や不足を整理する。
3. GPTが判断すべき科学条件を、曖昧さのない具体的な選択肢として提示する。

**新規科学計算は実施しません。**

## 1. 入力資料

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

最新公開HEAD：

`a9ee6a900a8e41fdb42126bfd01d20f933964b66`

固定source：

`4d35c2bc4b11a61dbb38f876e826a8d56c19f0ab`

主要資料：

- [Parent allocation技術実装準備・GPT引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9ee6a900a8e41fdb42126bfd01d20f933964b66/docs/tracks/algorithm_codesign/parent_allocation_source_preparation_and_gpt_handoff_20261014.md)
- [比較計画の技術整理v2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9ee6a900a8e41fdb42126bfd01d20f933964b66/docs/tracks/algorithm_codesign/parent_allocation_comparison_technical_options_v2_20261014.md)
- [Parent allocation技術監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c12558d682f6aef3cfc68e8d75082269f2282ac8/docs/tracks/algorithm_codesign/parent_allocation_technical_audit_20261014.md)
- [GPT研究方針レビュー原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md)

共通運用指示：

`docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md`

まず必要な資料をGitHubから取得し、固定commit・実際のsource・作業branchの状態を確認してください。

過去のレビュー・数値結果は必要に応じて参照しますが、全資料の再監査や既済テストの無目的な反復は不要です。

## 2. 作業A：Native実装との技術的な接続確認

現在のParent allocation sourceについて、既存のG9/native実装との接続を詳しく確認してください。

対象は以下です。

- `parent_allocation.py`
- `g9_p5.py`
- `g9_native.py`
- 既存のnative synthesis pipeline
- rotation key・cache管理
- strict operator phase guard
- finite-bit・angle誤差の既存処理

特に、以下の未解決事項を調べてください。

1. 旧cache schemaと新しいrotation key schemaの差
2. 旧cacheを安全に再利用できる条件
3. synthesis tool、runtime、seed、precision、source identityの取得方法
4. 新しいcontrolled rotationと既存native IRの対応
5. pure-child／pure-parentのexactなcontrolled位相処理
6. signed angleとadjoint sequenceの対応
7. 既存strict operator guardを新しいsourceへ接続する方法
8. native誤差をParent allocationのerror accountingへ戻す方法

**技術的に独立して解決できる互換性問題は、今回の作業で修正して構いません。**

ただし、登録済み科学入力を使うoperator validation、新しいnative合成、実際の科学資源評価は実施しないでください。

既存cacheについては、metadata・schema・hash・保存guardの読取り確認に限定してください。未確認の保存合成列を成功済みとして扱わないでください。

登録外のmockや人工的な小規模入力を用いたfocused testsは実施して構いません。

## 3. 作業B：最小比較計画の技術的な弱点の調査

現在の比較案では、主に次の配分が検討されています。

- 既存common-angle配分
- 一つのchildへparent質量を集中する頂点配分

しかし、既存の技術監査では、正の固定単価モデルにおける最適配分が、一般に内部解として得られることも示されています。

そのため、**頂点配分だけを比較すると、有利な内部配分を見落とす可能性があります。**

この点を技術的に整理してください。

具体的には、次を調べてください。

1. 固定単価モデルにおける内部最適配分の数学的な構成
2. 内部配分を比較候補へ追加する場合の必要情報
3. 実際の角度依存native価格では、固定単価の最適配分をそのまま使えない理由
4. 内部配分を導入した場合の新規synthesis key数の上限
5. 既存common-angle・頂点・内部配分の比較で区別できる改善機構
6. 内部候補を選ぶためのcost情報を、どこまで事前に必要とするか
7. 配分の最適化に使った情報と、性能を評価する情報を分離する必要性

数式の検証や登録外の形式的な解析は実施して構いません。

**登録済みD1〜D3の入力・費用表を用いた新しい最適配分の探索や、資源順位の再計算は行わないでください。**

今回の目的は、どの候補集合が科学的に適切かをCodexが決めることではなく、GPTがその判断を行うための技術的な根拠を揃えることです。

## 4. 作業C：Proposal設計の技術的整理

現在実装されているproposal候補について、数学的意味と実装条件を整理してください。

対象：

- canonical proposal
- root-local cost-aware IS
- global cost-aware IS
- regularized cost-aware IS

特に以下を確認してください。

- 各proposalが使用する情報
- normalizerとproposalの取得方法
- full-supportの保証
- zero-T-cost eventの扱い
- 正則化係数が何を制御するか
- 正則化がactual costを意味しないこと
- finite-bit後のmoment・rangeへの影響
- 必要なclassical setup・online sampling費用
- ordinaryとclosed P5へ同じ最適化機会を与える条件

**一般的なcost-aware ISの解析式と、有限confidenceでの真の最適proposalは区別してください。**

既存理論から確認できる事項と、新しい科学比較を行わなければ分からない事項を分離してください。

新しいproposalを科学的に採択したり、未承認の登録データで最適化したりしないでください。

## 5. 作業D：科学的未決定事項のDecision Packet作成

GPTが次のレビューで判断すべき内容を、**一つのDecision Packet**にまとめてください。

対象は以下の7領域です。

| 領域 | 整理する内容 |
|---|---|
| 1. 比較方式 | ordinary、closed P5、その他の対照の選択肢 |
| 2. 配分候補 | common-angle、頂点、内部配分の違い |
| 3. Proposal | canonical、root-local、global ISの違い |
| 4. 正則化・zero-cost | 各処理の技術的な成立条件 |
| 5. Accuracy・confidence・budget | 設定候補と比較の公平性への影響 |
| 6. Primary endpoint | T、CX、1Q、準備費用などの位置付け |
| 7. 科学的解釈 | 何が示され、何は示されない比較になるか |

各領域について、以下を記載してください。

- **選択肢**
- **既存資料によって確認できていること**
- **各選択肢の技術的な成立条件**
- **必要になる追加計算・合成・取得情報**
- **計算量・実装量・想定されるfailure**
- **比較の公平性に関する注意点**
- **科学的判断が必要な理由**

科学的な推奨・採択は行わず、必要であれば技術的な選択肢の優劣のみ説明してください。

例えば、あるproposalを採用すれば全event価格表が必要になる、という技術的帰結は明確にして構いません。

一方、「研究価値が最も高いから採択する」といった判断はGPTへ残してください。

## 6. 作業E：実行前の残存技術課題

現在のsourceを科学実行へ接続するために、何が必要かを整理してください。

ただし、今回の作業で科学worker全体を実装する必要はありません。

以下の区分を用いてください。

**A：科学条件の採択前に解決でき、今回解決した技術課題**

**B：科学条件が決まればCodexだけで実装できる課題**

**C：GPTによる科学的判断が先に必要な課題**

**D：新規科学計算の別途承認が必要な課題**

特に、production synthesis runtime、cache bridge、strict guard、error ledger、worker、failure handling、authorization、markerの依存関係を明確にしてください。

未取得のruntimeやtoolがある場合は、名称・版・取得方法・不足理由を報告してください。

ただし、未承認のnative合成や登録科学評価を起動してはいけません。

## 7. 作業上の制約

今回認可する作業は、既存sourceの静的調査、互換性に必要な限定的修正、登録外技術テスト、数理的な形式検証、比較条件の技術整理、Decision Packetの作成です。

以下は禁止します。

- 新しい科学計算
- D1〜D3の再実行・再ランキング
- 登録入力を使った新しい配分最適化
- 新しいnative T合成・compile
- 科学的な資源比較結果の生成
- 科学実験条件の独断による採択
- 新アルゴリズムの独立新規性・性能優位の認定
- 旧source・result・authorization・marker・STOPの変更
- 過去のclosureの解除
- Track A・N3への介入
- 論文原稿の作成

技術的な問題が見つかった場合、承認範囲内で解決できるものは解決してください。

科学条件の変更が必要なものは、問題・原因・解決案を整理し、GPTへ判断を戻してください。

既済の62テストや旧監査を、理由なく全面再実行しないでください。source変更時は影響範囲に応じたfocused regression testsを実施してください。

## 8. 成果物

独立branch/worktreeで作業し、必要なsource修正と資料をcommit・pushしてください。

成果物は以下に限定します。

1. 必要な互換性修正と登録外テスト
2. Native接続の技術的準備状況
3. 配分候補・proposal候補の技術的比較
4. **GPT向けDecision Packet**
5. 未解決課題・実行承認境界の整理

Decision Packetは、一つのMarkdownへまとめることを優先してください。

既存資料を単にコピーするのではなく、**GPTが7領域の科学的判断を行う際に、追加の技術調査を要求しなくてよい状態**を目指してください。

すでに必要な情報が揃っている項目については、無理に追加作業を作らず、既存の固定証拠を参照してください。

最終報告では、branch・commit SHA・公開URL・テスト結果・未解決事項を提示してください。

**完了後はmandatory STOPしてください。**

GPTによる詳細科学レビューは自動開始しません。ユーザーが明示的に承認した場合のみ開始します。

科学実行にも、別途ユーザーの明示承認が必要です。