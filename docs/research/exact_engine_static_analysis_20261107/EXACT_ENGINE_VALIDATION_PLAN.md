# Exact engine高速化：次回の最小技術検証案

2026-11-07 JST。**計画のみ。今回の実行許可は調査・Markdown作成・公開までで、下記の実装・test・benchmarkは未実行。**
候補の意味と保証区分は [OPTIONS](EXACT_ENGINE_OPTIMIZATION_OPTIONS.md)、現SOURCEと保存timingは [ANALYSIS](EXACT_ENGINE_HOTSPOT_ANALYSIS.md)。

## 1. 最初に試す範囲

第一候補はO1「exact零による疎/対角Fraction contraction」、続いてO2「ordered support別index計画」。**元dense kernelをoracleとして残し、O1単独、O1＋O2を比較する。** 新kernelを既存科学engineへ差し替えることは最初の技術検証に含めない。

| 項目 | 次回の最小内容 |
|---|---|
| 固定reuse SOURCE | SCV準備 `560c6092bfb5e71af8eedc4a4be5416922764f98` 内の `a1_engine_connection.py` の `ZERO,multiply,add,apply_exact,norm_squared,born_probability,threshold_count` |
| 最小実装 | 独立moduleの疎row生成・対角検出・候補contraction。次に独立index plan cacheを追加。dense fallbackと旧SOURCEは保持 |
| 人工test | 手書きの小さいexact行列/stateと固定word tape。実保存H4のstate/matrix/LAW/catalogは読み込まない |
| 初回で変更しないもの | `SavedEngine`、certificate/profile、sampler/facade、科学runner、価格cache、pool/manager/STOP、Born/threshold関数 |
| 必要な承認 | 新しい人工prototype・人工correctness・人工benchmarkを行う次回作業への明示指示。承認証明書・grant発行を新要件にしない |
| 成果 | 原kernelと候補のexact同値性、人工条件別のtime/memory/演算数、継続又は終了理由。実科学での速度を外挿しない |

今回の依頼をこの実装許可と解釈しない。既存52/72/106人工PASSも新候補のPASSへ読み替えない。

## 2. 再利用する資産と不足

全path gate順照合・named gate/matrix identity・支持qubit順・XY符号・正規化・threshold・変異拒否は既存testを再利用できる。一方、既存testのfloat `allclose`は独立参照として有用でも、**exact候補間の全振幅の有理数同値性**を代用しない。

| 既存test（固定SCV準備版） | 再利用する条件・目的 |
|---|---|
| `tests/tracks/algorithm_codesign/test_a1_engine_connection.py` | `test_exact_local_application_qubit_order_matches_statevector`、`test_exact_normalization_without_sqrt_and_scale_invariance`、`test_grid_endpoints_and_exact_threshold_count`、`test_runtime_rejects_uncertified_or_changed_gates` を基礎にする |
| 同上の人工DF fixture | gate順・parameter hex・support・global phaseと既存full builderの一致。初回の手書きmatrix testを通した後、人工builder接続を許可した範囲で利用 |
| `test_a1_n3_connection.py` | facadeのprivate境界、phase二重適用なし、runtime/input不一致、保存証明復元で再認証なし、priceとの同じXY回路とcache hit全額課金 |
| `test_a1_n3_campaign.py` | pinned word replay、bit先行保存、価格失敗時bit保全、STOP後の新probability拒否、groupのtrial順と全費用会計 |
| `test_scv224_pilot.py` | 将来新SOURCEへ接続する場合のfresh cache/source/STOP、固定quota、word/domain、最初の失敗でSTOP。現224-shotへ変更しない |
| `docs/research/o2_a1_engine_qualification_20261106/CORRECTNESS.json` | 保存値とraw hashの違い、signed-zero witnessの扱いをO3設計へ再利用。再認証ではない |

既存testsはSOURCEを読んで適合候補としただけで、今回は実行していない。初回は標準libraryと手書きexact入力を中心に、science loader・RNG・compiler・qualificationを呼ばない構成にする。

## 3. 人工correctnessの具体的な検査

### A. 局所contraction

- 行列：1/2/3-qubitのexact対角、identity、−1/±i、H型dense、xx_plus_yy型の2×2混合block、ordinary closed-control型、一般疎/dense。手書きFractionを使い、理想gateのsin/cosを新計算しない。
- state：零成分の多いもの、dense complex、非単位norm、相殺を含むdyadic、分母が異なる一般有理数。固定された小さい列挙集合を用い、科学PRNGを使わない。
- support：隣接・非隣接・逆順、ancillaがlocal LSB側に来る例、同じ集合でも順序が違う例。重複/範囲外、dimension/shape不一致は旧契約と同様に棄却する。
- 各gate作用後、全real/imag Fractionが元dense oracleと厳密に等しいこと。最終pだけで相殺されたバグを見逃さない。inputとcertified相当matrixが変異しないことも確認する。
- exact零だけを除く。極小の非零dyadic entryを入れ、削除しないことを検査する。diagonal fast pathの条件不成立時は疎又はdenseへ移る。

### B. index計画

旧式から生成したindex列と新planを完全比較し、全blockの非交差・全出力の一回更新を確認する。cache keyは `(N,ordered support)` とし、Nと順序を変えた場合を検査する。cold生成・warm再利用・eviction後再生成で結果が同じこと、planに外部から変異が入らないことを確認する。

### C. Born/measurementへの接続

候補stateを**旧** `born_probability` へ渡し、元dense stateからのpとFraction等値、`threshold_count`の整数等値を確認する。非単位norm・zero normのSTOP、p=0/1、`p=k/M`とその直前/直後、word `t−1/t`（合法範囲内）を含める。

同じ固定word tapeから同じbitを返し、pathとmeasurementのword消費/domain/ticketは不変。確率・state・threshold・真varianceをCV特徴量/学習/配分へ公開しない。bitの使い回しで高速化しない。

exact equalityはこの候補が**同じ保存有理数演算を実装する**という主張に必要な局所correctness条件であり、全研究へbitwise一致を一般要求するものではない。

### 同値性の説明を有限testと分けて残す

O1では各rowについて旧式 `Σ_j matrix[i,j]*v[j]` のexact零の項だけを取り除くため、同じ有理数値を返すことを代数的に示す。対角化もこの特殊例であり、tiny非零は残す。O2では旧index式と同じordered offsetsから同じ非交差blockを列挙することを示す。この局所同値性と入力不変性からgate列長に関する帰納法で全state列の同値性、同じ非零normからp/tの一致が従う。人工testはこの説明と実装の対応を検査するもので、有限testのPASSだけを全pathの証明へ格上げしない。

### D. O3以後へ進む場合だけ追加する検査

gate prebindは同じname/angleで異なるclass/definition/matrix、mutable objectの変更、signed-zero、証明hash/profile/shape不一致を拒否する。templateは人工全pathでgate名・angle hex・ordered support・phase・順序・cap・price readoutを比較する。prefix/path cacheはkey各項の不一致、state alias、fresh wordを二重/未消費にしないこと、call accountingを検査する。これらは初回O1/O2へ一括導入しない。

## 4. 人工性能測定

correctnessを先に通し、旧dense／O1／O1＋O2の三条件を同じ人工入力・同じ環境で比較する。入力作成と測定対象を分け、checksum/厳密比較に必要な時間をkernel時間と混ぜない。warm実行順の偏りは入替で確認し、repeat数はばらつきを識別できる程度に選び、無意味に長い実行を求めない。

| 測定対象 | 測定方法・区別 |
|---|---|
| 前処理 | 疎row作成、対角判定、index plan生成、cold bindを別timer |
| state適用 | `perf_counter` wall、単独processのCPU time、gate列全体と必要な関数別区間。人工入力でのみprofileを許可する案 |
| Born/threshold | 旧関数を別timer。O1/O2の速度へ混ぜず、全人工path wallも併記 |
| bit長・演算構造 | 人工stateの分子/分母bit長、zero/identity/nnz、gate supportと列長。実H4の分布と称しない |
| memory | plan/row cacheのentry数、tracemalloc等のPython allocation、process RSSの範囲・high-water。計測自体の負荷を分離 |
| cache条件 | cacheなし、cold一回、warm反復、eviction/missを区別。新しいprivate memory cacheのみ、旧SQLite/live cacheを開かない |
| 結果 | time分布と計測条件、absolute wall、memory、correctnessを一緒に示す。速度向上を事前に数値約束しない |

初回の単独process比較はkernel差を識別するためで、将来の科学worker数を少数へ固定する規則ではない。単体差が確認された場合のみ、同じ人工taskのprocess並列で起動/終了、serialization、forkの共有/私有memory、workerごとのcache、native threadを確認する。実際の空き資源・quota・非干渉に応じて並列度を選び、CPU/RAM/workerへ恣意的な低い上限を新設しない。

## 5. schedulingの独立した最小mock検証

必要なら別差分として、旧`Campaign.group`を再利用したmock pool/jobで、task数がworkersより少ない・等しい・一つ多い境界、短長task混在、同時完了、途中失敗を検査する。sleepで重いworkloadを作る必要はなく、仮想完了順を注入できる。

parentで旧順にclaim/path/word/ticketを確定してから投入だけ変える案なら、trialのword tape・bit・journal・最終quota・会計が不変か比較する。train/freeze/eval境界を越える先行実行とSTOP後の追加shotを禁止する。managerを作り直すこと、実224-shotへ接続することは含めない。

## 6. 科学SOURCEへ移す前の条件と承認境界

| 段階 | 許可対象に含める内容 | 含めない内容 |
|---|---|---|
| 今回 | 静的読解、既存値引用、4 Markdown公開・照合 | 全prototype/test/benchmark、科学計算 |
| 次の人工技術検証 | 独立O1/O2実装、手書きexact入力、人工同値性・time/memory計測。必要なら人工builder/mock lifecycle検査を明記 | 実H4 state/matrix投入、Born、実回路build/compile、sampling/measurement、旧qualification再実行 |
| 必要な実科学性能検証 | ユーザーの別明示承認後、対象SOURCE/input/path/axis/有限call・出力・STOPを定めた実H4旧新比較 | 旧752/224の再開、追加shot/campaign、保存結果の科学的再ランキング |
| 採用・新科学run | 同値性と保証の適合を確認した新SOURCE、新runの明示scope | 旧certificate/profileの上書きや現在の224-shot kernel差替え |

実科学比較が必要なら、保存H4/1.00 Å/STO-3G/legacy DF12/全4電子sector/`L_D3,m0/q1,δ.02,r1,K2,atol0,X/Y`、同一N1/N2/certified matrixとwordモデルを明記する。有限の事前指定pathについて旧新state/p/t比較とwall/memoryを別計測する案を、その時点で具体化する。実gate/circuit build回数、Born回数、compile0か必要回数、RNG0か固定tape使用を明示し、今回そのpath集合を実入力から抽出しない。engine実装の数学的同値性を、少数実pathの近似的一致だけで一様認証しない。

O1/O2がsame-matrix・same-order・same-state・exact p/tを保つことが確認できれば、新roundoff項は不要である可能性がある。保存δ*/Lmax/E/N2/grid証拠の適用前提を新SOURCEへ対応付ける。**旧artifactの書換えや、現hash lockを緩める対応はしない。** float/interval/mixed precisionへ進む場合はOPTIONSの追加保証を先に設計し、局所gateの保存証明だけで十分としない。

## 7. 継続・終了とmandatory STOP

- 全振幅/p/tの不一致、support/phase/word順の変更、identity/cache変異、未認証fallback、既存原本への書込みを検出した候補はSTOP。元kernelは維持し、原因を人工入力で切分ける。実科学retryで救済しない。
- correctnessが通り、人工の長いgate列でも改善が安定し、前処理とmemoryを含めて利得が残る場合だけ次段階を提案する。疎gateだけ速くdenseやcold条件が悪化するなら、適用条件を限定するか候補を終了する。普遍的な速度向上を要求しない。
- 内訳でgcd/大整数費が主に残る場合にO7/O8、build/hashが残る場合にO3/O4、prefix反復とmemory余裕が確認できる場合にO6を検討する。いずれも自動実装へ進まない。
- 速度が測定ばらつき内、memory増が利得を損なう、保証負担が目的に見合わない場合は終了又は保留。速度だけで区分3を優先しない。
- 技術検証結果を保存・公開・remote本文/blob/tree照合後STOP。実H4検証・新SCV/RPE runは別の明示許可まで開始しない。承認ID・receipt・新grantシステムを要求しない。

今回も4 Markdownの公開照合が終わり次第mandatory STOPし、この計画の実行へ進まない。
