# Track B：Parent allocationの近接先行研究との厳密対応調査

## 目的

Parent allocationについて、科学的設計レビュー前の最後の技術調査を実施してください。

目的は、**今回構成したParent allocationが、既存のEuler pairing、確率的合成、importance sampling、Taylor再配分などにどこまで含まれるかを、数式・アルゴリズム・情報アクセス・実装費用の各水準で明確にすること**です。

これまでの数学監査・source実装・native接続準備を繰り返す必要はありません。

今回の調査は、技術的な同値性・包含関係・具体的な差分の確認に限定します。

**独立新規性の最終認定、研究方針の採択、科学計算の実行は行いません。**

## 1. 入力資料

Repository：

`HIROMU1015/Partially-Randomized-Trotter`

最新公開HEAD：

`50c3db592041709247e5bd8f3d6849b6d20f3c10`

最初に以下を確認してください。

**最新Decision Packet**

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50c3db592041709247e5bd8f3d6849b6d20f3c10/docs/tracks/algorithm_codesign/parent_allocation_scientific_design_decision_packet_20261014.md

**Parent allocation技術監査**

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c12558d682f6aef3cfc68e8d75082269f2282ac8/docs/tracks/algorithm_codesign/parent_allocation_technical_audit_20261014.md

**GPT研究方針レビュー**

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md

共通運用指示：

`docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md`

過去のR0・G1・G5・G6・D1〜D3の先行研究照合記録を再利用してください。

既に十分確認された事項を無目的に再調査しないでください。

## 2. 特に確認する先行研究

優先する対象は以下です。

1. **Sparse Probabilistic Synthesis of Quantum Operations**
   - Koczor
   - https://arxiv.org/abs/2402.15550

2. **Quantum Simulation via Stochastic Combination of Unitaries**
   - Peetz, Smart, Narang
   - https://arxiv.org/abs/2407.21095

3. **Resource-Optimal Importance Sampling for Randomized Quantum Algorithms**
   - Cugini, Atif, Subaşı
   - https://arxiv.org/abs/2603.13495

4. **Wan–Berta–Campbellのrandomized time evolutionとEuler pairing**
   - https://arxiv.org/abs/2110.12071

5. **Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations**
   - https://arxiv.org/abs/2212.04566

必要な論文は、実際に取得できた版を明記し、対応する式・定理・アルゴリズムの位置を示してください。

既存の取得記録が十分であれば再利用して構いません。

網羅的な新規性調査を目的とするのではなく、**現在のParent allocationに最も近い既知構成との関係を確定できる範囲で確認**してください。

## 3. 今回の構成との対応調査

対象となるParent allocationは、

\[
z_i\ge0,\qquad \sum_i z_i=a,
\]

\[
c_i=\sqrt{z_i^2+b_i^2},
\qquad
\phi_i=\operatorname{atan2}(b_i,z_i)
\]

により、parentの係数をchild別に配分し、同一の有限operator meanを実現する構成です。

以下を区別して調べてください。

### A. 数学的な包含関係

- 既知の一般的なLCUやprobabilistic synthesisの枠組みに、この配分familyが含まれるか。
- 既存Euler pairingの自由度として既に表現できるか。
- R0・G1・G5の再配分と、変数変換によって完全に同じ問題になるか。
- cost-aware importance samplingだけで、同じ変更を再現できるか。
- 同じ平均演算子を得ることと、同じ乱択unitary ensembleを得ることを区別できるか。

単に一般的な線形結合の枠組みに含まれることと、具体的な生成手順が既知であることを混同しないでください。

### B. アルゴリズムとしての対応

以下を比較してください。

- 入力情報
- unitary辞書
- 係数の生成手順
- 回転角の決定
- sampling law
- importance weight
- normalization
- 有限bit処理
- controlled相対位相
- 古典的なsetup・online費用
- native回路費用

特に、一般的なSPSの最適化枠組みがParent allocationの配分を数学的に含む場合でも、**全辞書の明示生成を要求するのか、P5の少数群・root局所情報から生成できるのか**を分けて確認してください。

### C. 最適化問題としての対応

既存の解析結果

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2}
\]

と、その固定単価モデルでのKKT条件が、既知のimportance sampling・SPSの一般理論から直接得られるかを確認してください。

そのうえで、

- 固定されたunitary辞書に対するproposal最適化
- unitary辞書そのものを変える配分最適化
- 角度依存native費用を含む配分最適化
- 有限confidenceの総資源最適化

を明確に区別してください。

既知の一般原理から直接導ける部分を、独立した新定理として扱わないでください。

逆に、一般的な枠組みに包含されるという理由だけで、具体的な生成法・情報取得費用・実装上の追加効果まで同一だと判断しないでください。

### D. Controlled first momentとchannelの違い

SPSなどのprocess-levelの手法と、今回のcoherent first operator momentとの関係を明確にしてください。

特に、

- controlled unitaryへの埋め込み
- ancillaのoff-diagonal block
- 必要なoperator mean
- full quantum channelとしての一致
- global phaseとcontrolled relative phase

を区別してください。

「channelが異なるから比較不能」と短絡せず、適切な埋め込み・特殊化によって同じ問題を扱えるか確認してください。

## 4. 技術調査の判定

各先行研究について、次のように区分してください。

**A：明示的な同値性を確認**

同じ対象、情報、自由度、具体的アルゴリズムを対応させられる。

**B：一般理論には包含されるが、具体的な生成手順や情報アクセスに差が残る**

包含される部分と、追加手順が必要な部分を特定する。

**C：比較した限定条件では具体的な差分を確認**

どの仮定で異なるかを明示する。

**D：取得資料からは確定できない**

未確認の式・アルゴリズム・実装条件を特定する。

これらは技術的な分類です。

**「新規性が証明された」「新手法として採択する」といった科学的判断は行わないでください。**

完全同値が確認された場合、候補を救済するために別の自由度や新しいアルゴリズムを追加しないでください。

その事実と適用範囲を報告して停止してください。

## 5. 作業範囲と制限

今回許可するのは以下です。

- 既存論文・資料の取得
- 数式・アルゴリズムの静的比較
- 既知手法との対応式の導出
- 必要な限定的形式検証
- 既存sourceのread-only確認
- 技術的な未解決事項の整理

以下は禁止します。

- 新しい科学計算
- 登録D1〜D3入力の再評価
- Parent allocationの新しい資源最適化
- 新しいnative T合成・compile
- 新しいHamiltonian・科学用回路・trajectory生成
- 本番workerの起動
- 科学条件7領域の独断での採択
- 新規アルゴリズムの性能優位・独立新規性の認定
- 旧source・result・authorization・marker・STOPの変更
- Track A・N3への介入
- 論文原稿の作成

大規模なsource実装は行わないでください。

技術的な同値性を確認するために形式テストが本当に必要な場合のみ、登録外の小規模なテストを実施してください。

## 6. 成果物

原則として、以下をまとめた**1本のMarkdown技術報告**を作成してください。

1. 調査した先行研究と取得版
2. 対応する定理・式・アルゴリズムの位置
3. Parent allocationとの包含・同値・相違点
4. SPS・SCU・importance samplingとの関係
5. 既知の部分と、技術的な差分が残る部分
6. 未確認事項と、その理由
7. 次のGPT科学レビューに与える具体的な情報

特に最後の項目では、**今回の調査結果によって、既存の7領域の科学判断のどこが変わり得るか**を整理してください。

ただし、7領域そのものをCodexが採択・変更してはいけません。

既存Decision Packetへの参照追加は必要最小限に限定してください。

独立branch/worktreeで作業し、結果をcommit・pushしてください。

最終報告にはbranch、commit SHA、GitHub URL、取得できた資料、技術分類、未解決事項を記載してください。

**完了後はmandatory STOPしてください。**

GPTによる科学レビューは、ユーザーの明示的な開始承認を受けるまで開始しません。

新しい科学計算も別途の明示承認まで実行しません。