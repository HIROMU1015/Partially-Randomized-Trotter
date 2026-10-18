# 新研究候補：1件のみを条件付きで残す

[入口](README.md)／[過去の終了理由](existing_research.md)／[最小検証案](minimal_verification.md)／[証拠](evidence_index.json)。本書の「提案・推論」は今回の技術的整理であり、定理・保証・新規性の認定ではない。

## 優先表

| 優先 | 研究課題 | 改善の具体的機構として調べるもの | 現在の評価 |
|---|---|---|---|
| 1（唯一、条件付き） | 固定粒子数・DF構造を保持した全エネルギーshadowsの共分散縮約 | DF係数の積とorbital-rotation共分散のcharge/incidence classを、全係数・全共分散表の展開前に縮約し、相殺と粒子数制約を保持する | **方法差未確定・新生成則未構成・性能未評価**。最初は紙上の包含・計算量比較だけを検討する |

第2・第3候補は提案しない。一般測定法、MPS準備、算術・encoding、PF/THRIFTの再包装は[選別表](existing_research.md)で除外した。唯一の候補も「今すぐ実装すべき有望新法」と認定したものではない。

## 1. 何を改善する問いか

与えられた粒子数ηのstate copiesから、既知の電子Hamiltonianのenergyを推定する測定・古典後処理の費用を対象とする。候補の第一目的は**共分散取得・保存・縮約の古典費用**であり、そこから十分shot上界や総費用も改善できるかは別の未解決段階とする。coherent時間発展・RPEの直接改良とは主張しない。

概念的な表現は、normal-order補正とscalarを保持した

\[
H_{\rm DF}=cI+Q+\sum_{\ell=1}^{R}w_\ell A_\ell^2,
\quad Q=\sum_{pq}h_{pq}a_p^\dagger a_q,
\quad A_\ell=\sum_{pq}(L_\ell)_{pq}a_p^\dagger a_q
\]

である。これは既知DF表現の記法であり、新しい分割・PSD化・shiftを提案していない。Rや行列rankが小さいという条件を明示し、一般dense化学Hで自動的に満たされるとはしない。分子・geometry・basis・rank policy・許容誤差は未選定であり、今回保存H4へ代入していない。

## 2. 中心的なアイデアと適用条件

既知orbital-rotation ensembleから得た1-/2-RDM推定量を \(\widehat D_\mu\)、Hamiltonianの対応する実Hermitian基底係数を \(v_\mu\) と書く。定数を別扱いにした全energy推定量の分散は、通常の共分散恒等式

\[
\widehat E=c+\sum_\mu v_\mu\widehat D_\mu,\qquad
\operatorname{Var}_\rho(\widehat E)
=\sum_{\mu\nu}v_\mu v_\nu
\operatorname{Cov}_\rho(\widehat D_\mu,\widehat D_\nu)
\]

で表される（複素表示では共役・実部を対応させる）。この恒等式と共分散自体は新しくない。

**今回の仮説：** 既知の共分散classへDF係数をfactorのまま先に縮約し、必要なstate correlationの線形結合も圧縮できれば、全 \(O(n^4)\) quartic係数やその対を明示保存する方式より古典費用を下げられる可能性がある。charge選択則・fermionの反対称性・mixed quadratic/quartic項・block間相殺を、項別絶対値和へ置き換える前に保持することが機構である。

ただし **\(O(n^8)\) は四つ組係数対を愚直に列挙する場合のindex数であり、最強既知法の必須計算量ではない**。既知法もclass分類・streaming・cache・DF・tensor contractionを使用できる。その強対照を超える操作数・storage・state情報取得の差を示せなければ、提案は既知法の実装上の整理へ還元される。具体的な縮約schedule・新しい漸近上界・実測speedupは今回得ていない。

既知測定法を固定して正確に同じ分散を速く計算するだけなら、物理的な分散・shots・量子gate数は変わらない。従来より鋭い十分shot boundが得られることと、既知法の実際の必要shotsを減らすことも区別する。shot削減を主張するには、強い既知対照にも同じ鋭い共分散情報を与えたうえで別の測定/estimator差が必要であり、その差は現時点で未提示である。

## 3. 主要先行研究との具体的な差分

| 文献・固定版 | 既に可能なこと／今回確認した箇所 | 候補へ残る問いと重複リスク |
|---|---|---|
| [Low, 2208.08964v2](https://arxiv.org/html/2208.08964v2) | 粒子数保存random orbital測定、逆channel、k-RDM estimator、平均分散。§§III–V | 粒子数保存やquartic期待値の不偏推定を新規性にできない。平均RDM保証を任意係数全energyのworst-case保証へ移さない |
| [Westほか, 2606.27254v1](https://arxiv.org/html/2606.27254v1) | Slater overlapのworst-case bound、quadratic particle-preserving observable。ResultsとApp.C | quadraticのmode依存改善をそのままquartic DF全energyへ適用できない。第一量子化designの回路費用も別問題 |
| [Koizumiほか, 2606.30601v2](https://arxiv.org/html/2606.30601v2) | k-RDMの保証、1-/2-RDM・mixed共分散のcharge/incidence分類、化学energy derivative応用。§III、S6.3、S8 | **最も強い直接対照。共分散・化学への応用・class圧縮も既知**。DF factorからの全energy縮約がそのroutineの通常の使用に含まれるなら独立差なし |
| [Westほか, 2609.40173v1](https://arxiv.org/html/2609.40173v1) | homogeneous Majorana degree observable全体のmatchgate分散bound、mixed degreeへの拡張。Thm.1とその後、§IV | 個別monomialしか既知でないという前提は誤り。operator normと取得費、fixed-N版との条件差を含め対照に入れる。論文のHubbard改善率を本研究へ転用しない |
| [Majsakほか, 2402.19230v3](https://arxiv.org/html/2402.19230v3)、[McNultyほか, 2402.19349v2](https://arxiv.org/html/2402.19349v2) | quadratic/quartic joint fermion測定、variance・layoutの保証 | Haar orbital測定より安いcircuitの強い候補。per-copy測定費用を無視してqueryだけで勝ったとしない |
| [Choiほか, FFF 2208.14490v3](https://arxiv.org/html/2208.14490v3) | 一体項再配分、proxy state共分散によるfragment/測定配分最適化。§§2.1–2.3 | variance最適化やproxy/covariance利用自体は既知。Hamiltonian分割を改変する提案ではないが、同一copy・精度のstrong baselineとして必要 |
| [Hadfieldほか, 2006.15788v1](https://arxiv.org/html/2006.15788v1)、[Hadfield, 2105.12207v1](https://arxiv.org/html/2105.12207v1) | coefficient-aware局所bias、adaptive Pauli shadows | 係数依存測定の一般論を新規性にしない。pilot情報・training copiesと同じ最適化機会を両側へ与える |

本文の確認箇所と原本bytes/SHAは [調査記録](literature_scope.json)・[索引](evidence_index.json)に示す。新規性の最終認定、先行性の証明、論文化判断は行わない。

## 4. 必要な入力と無料扱いできない費用

- 元電子Hamiltonianの積分・核反発/凍結軌道定数・mode順・η/sector。DFを近似するなら元Hとの差の保証または評価範囲を別途明示する。
- stateの供給モデル。exact ground copiesなら条件付きタスクとし、実際のground preparationの解決としない。近似stateの \(\operatorname{Tr}(\rho H)-E_0\) は測定誤差とは別で、未知ならground-energy保証も未取得。
- 相関情報。quartic推定量の共分散は一般に1-/2-RDMだけで閉じると仮定しない。著者SOURCEはstateから必要なRDMを評価する構成を含む。高次RDM・近似proxy・測定pilotの取得費と誤差を明記する。SlaterのWick計算を相関groundへ無条件移送しない。
- 古典費用：DF生成/検証、係数classの生成・縮約、state情報取得、精度/bit長・storage・per-shot後処理。cacheの再利用数も両側共通にする。
- 量子費用：copy準備、Gaussian/Givens回転・routing、測定、rotation合成/FT費、training・失敗試行。measurement-only法をRPEのcoherent accessと同価格にしない。

主指標候補は共分散処理の操作数・peak storage・古典時間である。shot、RZ/CX/T・depth、総資源は接続先であり、改善を実証していない。1/ε²型copy測定とHeisenberg型coherent推定の異なるアクセス条件も残す。confidence/精度/入力classは未採択。

## 5. 実装難度・再利用・順位の理由

理論的難度は高い。非可換block間のmixed covariance、state依存高次相関、同情報の最強既知縮約との比較が主障害である。既存DFデータ構造・sector/mode/scalar identity・費用項の分類は再利用可能だが、PF/RPE runnerをshadows validatorと呼び替えることはできない。

公開SOURCE `koizumiyuki/reprod_ORshadow_covariances` のcommit `206aa441d6a08c1984975fddf14a23ccf054f370` は、1-/2-body estimator、共分散classの係数生成、state依存評価を実装する。READMEと関連関数を静的読取した。実行・import・依存導入・著者のPASS再現はしていない。complete tableのη/holes条件やPython/SymPy依存も記録済み。既存SOURCEにDF全energyの独立した縮約generatorを確認できなかったことは、文献上の不存在の証明ではない。

この候補を唯一残した理由は、D1内で元Hの直接energy読出しへ接続可能であり、旧A2の相対PF剰余や旧encoding案とは異なる処理対象で、最も近い先行式と公開SOURCEを固定できたためである。**新規性・成功可能性の確信度は低い**。まず同情報の既知式へ還元されるかを判別できるので、分子走査やcompileへの投資を先送りできる。
