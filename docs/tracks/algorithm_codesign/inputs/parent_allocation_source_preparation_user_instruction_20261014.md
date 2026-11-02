# Track B：Parent allocationの技術実装準備・独立検証

## 目的

Return aggregationのparent係数をchild別に配分する新しい構成について、既存の技術監査を引き継ぎ、**GPTによる次の重要科学レビューに必要な技術的準備を、可能な限りCodex側で完結**してください。

今回の目的は、新アルゴリズムの科学的採択や資源優位の認定ではありません。

**既に数学的に監査した構成を、正しく実装・検証できる状態まで具体化し、科学的判断が必要な未解決事項だけを明確にしてGPTへ返すこと**が目的です。

論文化・原稿作成は行いません。

## 1. 必ず確認する資料

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

最新の監査commit：

`c12558d682f6aef3cfc68e8d75082269f2282ac8`

まず、以下の資料をGitHubから取得して確認してください。

**① 技術監査報告**

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c12558d682f6aef3cfc68e8d75082269f2282ac8/docs/tracks/algorithm_codesign/parent_allocation_technical_audit_20261014.md

**② 最小比較計画案**

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c12558d682f6aef3cfc68e8d75082269f2282ac8/docs/tracks/algorithm_codesign/parent_allocation_minimal_comparison_draft_20261014.md

**③ 直前のGPT研究方針レビュー**

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md

必要に応じてG1〜G10、D1〜D3、R0、G5、既存P5 source、native実装、先行研究照合資料を確認してください。

共通運用指示：

`docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md`

旧版と矛盾する場合は最新版を優先してください。

## 2. 作業A：Parent allocationの実装準備

監査済みの数学的構成を、既存のP5実装に接続するためのsourceを準備してください。

対象は以下の配分です。

\[
z_i\ge0,\qquad \sum_i z_i=a,
\]

\[
c_i=\sqrt{z_i^2+b_i^2},
\qquad
\phi_i=\operatorname{atan2}(b_i,z_i).
\]

満たすべき平均演算子の恒等式は、監査報告の定義に従ってください。

実装上、少なくとも以下を扱ってください。

- root parent係数とchild別係数の取得
- 配分 \(z_i\) の非負性と総和の厳密な確認
- child別の係数・回転角・抽出確率の生成
- root群のnormalizer更新
- group lawとchild lawの整合
- actual proposalと補正weightの整合
- controlled相対位相
- 演算子の積順序と実際の回路時間順序
- 既存common-angle配分への正確な復帰

特に、監査で未対応とされた以下の端点を適切に扱ってください。

1. \(z_i=0,\ b_i>0\)：pure-child
2. \(b_i=0,\ z_i>0\)：pure-parent
3. \(z_i=b_i=0\)：zero event
4. \(s=0\)：純parent
5. \(a=0\)：配分自由度なし

端点を巨大な有限ratioで代用しないでください。

新しいroot群の係数が複数平方根の和になることにも注意し、既存の単一radicand前提をそのまま流用しないでください。

既存の非root部分と、既存P5の再現可能性を保持してください。

科学的な比較方式をまだ確定していないため、**任意の適格配分を扱える汎用的な実装を優先**してください。草案にある特定の頂点配分だけにハードコードしないでください。

## 3. 作業B：Ordinary側への公平な接続

新しい係数配分をclosed P5だけに適用すると、比較対象の自由度が不公平になる可能性があります。

そこで、ordinary側の対応するrootにも同じ配分自由度を適用できるよう、実装可能性を確認してください。

技術的に可能であれば、共通の係数配分・digital law・端点処理を使い、ordinaryとclosed P5の双方を表現できるようにしてください。

ただし、以下を混同しないでください。

- 配分を変更した効果
- Return aggregationそのものの効果
- proposal変更の効果
- 回転合成方法の効果

これらを区別できるsource構成としてください。

この段階で比較結果や優劣を計算する必要はありません。

## 4. 作業C：Finite-bit・位相・誤差会計の技術検証

以下について、実装上の整合性を確認してください。

**確率・係数**

- exact rational係数
- 平方根の区間評価
- dyadic proposal
- full support
- zero coefficientの除外
- 実際のproposalに対する補正weight
- proposal確率和
- ideal係数とdigital係数の違い

**量子演算子**

- 非可換な積順序
- 正負時間の形式的な意味論
- controlled相対位相
- pure-childとpure-parentの位相
- adjoint・逆回転の扱い
- 元の平均演算子の保存

**資源評価の入力**

- moment
- range
- acceptance
- coefficient error
- angle error
- native synthesis error
- classical setup cost
- online sampling cost
- 新角度の取得費用

これらの値を計算・検証する技術的な仕組みを整理してください。

ただし、未採択の科学実験条件を用いた新しい資源表や十分shot数を生成してはいけません。

## 5. 作業D：登録外の独立テスト

実装したsourceについて、既存の登録科学入力を使わない小規模な技術テストを実施してください。

重点項目：

- 数式どおりの平均保存
- 非可換な作用順序
- controlled位相
- 全端点ケース
- 既存common-angle配分への復帰
- ordinaryとclosed P5の共通adapter
- digital係数と実proposalの整合
- 不正入力の拒否
- 既存sourceとの後方互換性

可能な限り、既存のproduction evaluatorとは独立した方法で検証してください。

テスト件数だけを根拠に数学的証明が完了したと主張しないでください。

**登録済みのD1〜D3入力、新しいnative合成、科学runner、科学資源評価を実行しないでください。**

小規模な登録外ユニットテストや静的検証に限定してください。

## 6. 作業E：最小比較計画案の技術的完成

次の草案を引き継ぎ、技術的に実行可能な計画へ整理してください。

`parent_allocation_minimal_comparison_draft_20261014.md`

特に以下を調べてください。

1. 各比較方式に同じ自由度を与えられるか
2. canonical proposalとcost-aware proposalを正しく構成できるか
3. T費用0のeventで既知の最適IS式がどう扱われるか
4. 正則化proposalの実装上の必要性
5. 新角度の合成keyが何件必要になるか
6. 既存cacheを安全に再利用できる条件
7. endpointに必要なexact回路
8. 比較用sourceの依存関係
9. 既存の登録済み結果を変更せずに新しいledgerを作れるか
10. 実行前に不足しているtool・runtime・source identity

ただし、以下は**科学的な未決定事項**です。

- 比較対象の最終採択
- proposal候補の最終選定
- 正則化係数の採択
- accuracy・confidence・budgetの最終固定
- 何をprimary endpointとするか
- どの結果なら次の研究へ進むか

これらをCodexが独自に決定しないでください。

複数案がある場合は、技術的な長所・短所と判断に必要な情報を整理してください。

**科学的に未採択の条件を、既に承認された実行契約へ変換してはいけません。**

## 7. 作業F：実行前preflightと未解決事項の整理

ここまでの作業を踏まえ、技術準備状況を整理してください。

以下を明確にしてください。

- 実装済みの機能
- テストで確認した範囲
- 未実装・未検証の機能
- 既存sourceから変更した箇所
- 科学実行前に必要な追加実装
- 科学実行前に必要な取得・合成
- 想定する技術的な失敗条件
- source・runtime・toolの再現性
- GPTによる判断が本当に必要な事項
- ユーザーの別途計算実行承認が必要な事項

必要であれば、実行を伴わないpreflightやzero-compute guardを実装してください。

ただし、科学runnerの実行、登録資源値の生成、結果の先取りは禁止します。

## 8. 実施上の制約

今回認可するのは、技術実装準備、静的調査、登録外の技術テスト、計画案の整備のみです。

**禁止事項**

- 新しい科学計算
- D1〜D3の再実行・再ランキング
- 登録入力を用いた新しい係数・資源評価
- 新しいnative T合成
- 新しいHamiltonian・科学用回路・trajectory生成
- 実験条件の結果依存変更
- 比較計画の独断での科学的採択
- 新アルゴリズムの新規性・優位性の認定
- 既存authorization・markerの再利用
- 旧結果・closure・STOPの変更
- 論文原稿の作成

既存のTrack A・N3、親worktreeの変更も保護してください。

作業は独立branch/worktreeで行い、既存の公開済み科学成果を変更しないでください。

技術上の問題が発見された場合は、原因と対処案を調べ、承認済みの技術作業の範囲で解決してください。ただし、科学条件の変更が必要な問題を無断で解決してはいけません。

## 9. 成果物

作業完了後、以下をGitHubへcommit・pushしてください。

1. 新しいParent allocation用adapter/source
2. 登録外テストと実行結果
3. source・runtime・依存関係の確認記録
4. 最小比較計画案の技術的な改訂版
5. 実装済み事項・未解決事項・承認境界をまとめたGPT向け引継ぎ報告

資料は必要最小限のファイル数に整理してください。

GPT向け報告では、以下の分類を明確にしてください。

- `TECHNICALLY_READY_FOR_SCIENTIFIC_DESIGN_REVIEW`
- `TECHNICAL_PREPARATION_INCOMPLETE`
- `TECHNICAL_BLOCKER_REQUIRES_SCIENTIFIC_DECISION`

これらは技術的な状態分類であり、科学的GOや実行承認ではありません。

最後に、branch、commit SHA、GitHub URL、テスト結果、保護対象の確認結果を報告してください。

**作業完了後はmandatory STOPしてください。**

GPTによる詳細科学レビューは自動開始せず、ユーザーが明示的に開始を承認した場合にのみ実施します。

科学計算も別途の明示承認まで開始しません。