# O2新機構の技術的選別

2026-11-08。主文書は[NEW_MECHANISM_CANDIDATES.md](NEW_MECHANISM_CANDIDATES.md)。性能順位や独立新規性を認定せず、操作・必要情報・強い対照・次の有限判別材料で選別した。

## 残した候補と除外した方向

| 方向 | 新しい操作/情報 | 数学的な削減経路 | 主な障害・強い対照 | 今回の扱い |
|---|---|---|---|---|
| E：coherent label消去後の早期終了 | 正確率の小pool、SELECT後flagを共通suffix前に読む。失敗0、成功だけsuffix | ideal全attempt M2=p_s≤1、失敗時のsuffix費省略。flagだけの校正はsuffix不要 | 有料PREP/SELECT、p_s≥1/2、小角度の小利益。既知LCUにも同じ早期終了を認める | KNOWN_METHOD_APPLICATION + 限定技術候補。次のinstrument/全費用の小さい確認へ提案 |
| Gaussian product-word return | single-particle group積をreturn記述にする | 多数のGaussian productsを一つへ再合成できる場合 | 既知Gaussian fusionと同値になり得る。non-Gaussian barrier、数値合成費、価格normalizer未解決 | 独立O2候補から除外。O3へ移行しない |
| joint X/Y・system shadowsの追加 | 一回で複数のobservable情報を得る | 別のobservableを同時に知る用途はある | 同じ一つのcomplex平均へのN利益は自動ではなく、既知法。余計な情報の取得費がある | 新しい改善条件を特定せず、独立推奨なし |
| metadata/QMC/antitheticのみ | classical path取得を相関させる | path間のMonte Carlo成分へ作用し得る | 新しい物理的couplingなしではHadamard内noiseが残り、既知層別/QMCに還元。停止保証/費用も必要 | 既存法のパラメータ変更だけを候補化しない |
| sector恒等式によるlaw交換 | conserved chargeに関するnull成分の再配分 | 理想operator norm/envelopeを下げる場合がある | 固定finite-grid Q・selected gate targetの同一性を維持できる構造を未特定 | PROMISING_BUT_INCOMPLETEだが今回は推奨なし。targetを勝手に変えない |
| 全path価格に比例するproposal | 小summaryからconditional重みを得る | 安いnormalizerがあればsetup削減 | 今回もその具体構造なし。既存Parent/J1と同じ不足 | REDUCES_TO_EXISTING_O2、調査を繰り返さない |
| Mの別slot/basis、Dの残差配置 | 既存候補の適用先変更 | 以前と同じ機構 | ユーザーが保留を指定。新しい取得構造なし | 再開しない |

「新しいrepository上の操作」と「世界で新しいアルゴリズム」を分ける。案Eが既知LCU/動的postselectionの同じ設計に完全に含まれる可能性は残り、その場合は既知法の応用として比較する。学術的新規性を理由に強いbaselineを除かない。[既知LCU](https://arxiv.org/html/1202.5822)、[randomized composite LCU](https://arxiv.org/html/2506.15658v1)、[Hadamardの追加情報利用](https://arxiv.org/html/2505.15913v1)

## 案Eの実装アクセスと未接続部分

最小の明示アクセスは、二つのsystem-only W_j、共通A/B、exact actual p、signal ancillaと一つのpool ancillaである。coherent PRNG、全path superposition、inverse-state-preparation、未知成功率・overlapは入力にしない。πからの外側取得とpool質量取得も有料で、表の全supportを無料で列挙する前提は置かない。

`PREP`のbit-grid/角精度、signal付きSELECTの制御位相、flagの実測/feedback、条件付きsuffixの価格、三値outcomeの保存・facade、prefix-only校正が未実装/未認証である。現在のA2 adapterへzetaを替えるだけでは得られない。A2のscale/decoderを流用すると別target/別M2になり得る。

## SOURCE再利用地図

以下はbase e98589a内の静的SOURCEを確認した範囲。実装・旧test・実データ処理を起動したという意味ではない。

| SOURCE | 使えるもの | 変更せずには使えないもの |
|---|---|---|
| `src/trottertracks/algorithm_codesign/a2_lcu_access.py` | 明示system circuit検査、phaseを保持するcontrol、PREP/SELECT/readout規約 | augmented state、β² decoder、fine−coarse target。新しいflag timing/decoderが要る |
| `a1_control_variates.py`, `a1_n3_connection.py` | actual CategoricalLaw、conditional取得、独立stage/stream、bounded校正/会計 | 現facadeは±1 bitと実pathを返す。三値flag/bitとcoherent poolへ無変更で流用しない |
| `m_label_average.py` | phase/ordering、additional readoutを有料とする規約、人工/engineの隔離 | common-V instrumentとeligibilityをEの選別器へ転用しない |
| `m_n0_header_binding.py`, `a1_engine_connection.py` | 共通prefix/suffixのapplication順とselected W意味論の参照 | 実N1へのpool再分割、group再集計を今回起動しない。M binderをEの科学runnerにしない |
| `src/trotterlib/rte.py`, `df_rte_qiskit.py`, `df_partial_s2.py` | order/sign/event phase、signed rotation、system gate記述 | 新しい二重制御/flag/動的suffixの誤差・価格は旧仕様外 |
| `g8_pipeline.py`, `rte_development/accounting.py` | 古典事前zero、all-trial分母、spent prefixやmiss費を残す原則 | quantum flagはBornを伴う有料取得。事前zeroと混同しない |
| `return_aggregation.py`, `g9_p5.py`, `parent_allocation.py` | finite-bit/weight/positive support、強いClosed/Parent対照 | coherent probability amplitude・全path価格normalizerは提供しない |
| `src/trotterlib/df_rpe_hadamard_compiled_cost.py` | full no-prep価格接口とprofile/cacheの参照 | dynamic abortと実現prefix費・期待suffix費の価格評価は別の接続が要る |
| `rte_development/boundary.py`, `src/trotterlib/df_trotter/ops.py` | D2のliteral inverse、既知Gaussian/対角primitiveの範囲 | Gaussian fusionを新しいO2 normalizerの証拠にしない |

新しい一般frameworkやscientific runnerを今回作っていない。必要なら次段階に一つのpair専用adapterを追加するが、既存SOURCE/科学契約はそのまま保持する。

## 強いbaselineを含む受理前条件

通常PRだけでなく、同じ外側層別・finite Q・supportに対するB1-OL/可能な細分配分、既知単独label直接測定/commuting・unitary grouping、同じpairの標準LCU＋早期flag、Closed/CTSを適用できる所ではその最適化を含める。既知の反対符号・global phaseだけのpairなら、古典的なoperator集約で無料/低費用に処理できることがあり、その例をEの利益の根拠にしない。

Qiskit dynamic制御や追加ancilla、state再準備を候補だけ除外しない。M2やflag校正の利益と、full-wrapperの制御/flag費の増加を別座標で比べる。旧価格・shot数・engine誤差を流用しない。実測分散/点費用を期待総資源保証にしない。

現時点の判断は、案Eの**有限の技術判別は可能**というもの。強いbaselineより安い、実H4で成功する、科学的GOに相当する、という判断はしていない。
