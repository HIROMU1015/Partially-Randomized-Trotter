# Track A AX-2B H4 v5後：独立科学レビューとH6への進行判断

- **作成日**：2026-10-10（JST）
- **文書区分**：ユーザーの明示的な開始承認を受けて実施したGPT独立科学レビューの記録
- **対象研究**：Partially Randomized Trotter（PR）、Track A、AX-2B H4技術pilot v5後
- **対象repository**：`HIROMU1015/Partially-Randomized-Trotter`
- **レビュー固定ref**：`d6510db9326e9335bedd03d0d07c490561e23112`
- **公開branch**：`track-a-ax2b-h4-post-review-20261010`
- **本書の権限**：科学的な評価・進行方針の判断。repository変更、分子計算、H6入力生成、sampling、compile、launchの実行指示ではない。
- **維持する実行状態**：H6は`H6_NOT_AUTHORIZED`。既存draft・terminal・freezeを書き換えない。

> **結論**：H4 v5は、登録範囲の実装整合性とbounded実行可能性を示す技術pilotとして受け入れる。ただし、総数値誤差の保証、要求精度への適格性、測定込み資源の優位は未確定である。RQ-Rを主軸、RQ-P1を補助とする研究方針は維持する。次はCodexで、H4の限定的な数値・estimator接続検証と、H6の7 cell・36 wrapper案の修正・実装準備をまとめて進める。H6技術pilotは総uの厳密認定を必須にしない限定目的で設計してよいが、入力policy・source・計算上限を固定し、別途明示的な実行指示を受けるまで起動しない。

## 0. 読み方と証拠の区分

本書では次の三種類を区別する。

**保存事実**は、repositoryの固定refに保存された結果、source、契約、監査が実際に記載・実装している内容である。**レビュー上の解釈・導出**は、その記録から行った数学的・科学的な検討である。**今後の提案・判断**は、まだ実施していない検証の目的、優先順位、進行条件である。

Codexの3文書[S01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_post_scientific_review_v1.md)[S02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_validation_gaps_v1.md)[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)はレビュー対象となる案であり、本書の独立判断とは別である。報告されたPASSをすべて独立再実行したという意味ではない。sourceの静的読解、一次JSONと報告の照合、一次文献との対応確認を行った。新しい分子Hamiltonian・stateの生成、既存runnerの実行、trajectory sampling、回路build/compile、test再実行、再fitは行っていない。保存値の比率・平方・複素ノルムなどの単純な算術は本レビューで計算し、そのことを明示した。

参照ID[Sxx]は末尾の固定commitリンク、[Exx]は外部一次資料を示す。旧レビューを参照するときも、その当時のscopeと現在のH4 v5のscopeを混同しない。

---

## 1. 何を判断するレビューか

今回の問いは、H4の実装が完走したかという一点だけではない。

1. H4 v5の技術検査は、どの作用・状態・回路・数値に対する整合性を支持するか。
2. 数値不確かさや測定込み会計の不足は、研究の継続を妨げる問題か、次に埋める限定的な問題か。
3. 不足検証11項目のうち、H6技術pilot前に必要な項目と、本格資源比較・H8独立評価までに必要な項目はどれか。
4. H6の7 cell・36 wrapper案は、その技術目的に対して過不足がないか。
5. 研究の主軸、次の担当、実行を止める条件をどう定めるか。

本レビューでは、H6本検証・H8独立評価を現時点で認可せず、H4 v5の費用からPRの一般的優位や化学的精度のエネルギー推定を結論しない。

## 2. 最終判定の一覧

以下は本書の判断表であり、既存machine-readable statusを変更するものではない。

| 判断対象 | 独立レビューの結論 | 残る条件 |
|---|---|---|
| H4 v5の技術的完了 | 登録scopeで受け入れる | 独立再実行・全空間数値証明ではない |
| H4 signalの数値的整合性 | 複数経路の一致を支持する | 共通入力・共通matvecの誤り、総roundoff上界は別 |
| H4の精度適格性認定 | 未認定を維持 | uの根拠とestimator接続が必要 |
| H4の期待費用・shot込みwinner | 判定しない | cost coverage、標本数、u-aware会計が不足 |
| RQ-R／RQ-P1の分担 | RQ-R主、RQ-P1補助を維持 | 今回の5 cost cellでFEWを再fitしない |
| H6 7 cell・36 wrapperの構成 | 技術pilot案として修正条件付き採用を推奨 | DF rankの暗黙config、数値scope、実行上限を修正・固定 |
| 厳密u未認定でのH6技術pilot | 限定目的なら許容する方針 | accuracyはUNDETERMINED、N/Gはnull、実行認可は別 |
| H6入力生成・科学launch | 今回は認可しない | 新source/input/環境/資源の固定と明示指示 |
| 次の担当 | Codex | 下記の限定検証・準備をまとめて担当 |
| 次の本格GPT判断 | H4限定検証とH6技術pilotの結果後 | H6本検証への科学的GO/STOP |

重要なのは「研究を継続する」「技術pilotの設計を採用する」「実際に計算を起動する」を別の判断として扱うことである。前二者の科学的判断をここで行い、最後の実行許可は与えない。

## 3. 来歴とレビュー可能性

### 3.1 実行時sourceと公開source

| 種類 | commitまたは状態 |
|---|---|
| H4 v5実行時base HEAD | `b2e1bf65e21893b6c617223b42313623d3186f12` |
| 実際の実行source | base HEADと、実行時にhash固定されたlocal source bytesの組合せ |
| H4一次結果・保存監査の公開 | `aa9b4768819680600d99ceb962b22aec16b99fb0` |
| レビュー文書・索引の公開 | `bda245df2f3165083f1de29ab052694773a44ac9` |
| 必須source24件の事後公開 | `31cdff47f3282d2898c153a1af22f90e500be4c6` |
| 旧監査source16件の公開 | `3e9358421c072af970a9a37025e7fdc9c51ce8ea` |
| 最新索引・公開照合記録／レビューref | `d6510db9326e9335bedd03d0d07c490561e23112` |

163 freeze項目（science157・validation6）の公開対応が記録され、代表sourceを含め必要な内容を取得できた。163件すべてのSHA-256をこのレビューで独立再計算したとは主張しない。[S19](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_gpt_review_index_v1.md)[S20](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_completion_v1/source_publication_manifest_v1.json)[S21](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/source_freeze_v5.json)

**事後公開によりsource監査が可能になったことと、実行時からGit commitだけでsource一式が固定されていたことは同じではない。** 公開commitを実行時commitへ読み替えない。旧結果のhashが保存されていることは証拠の保全に有用だが、外部CIまたは独立再現の代用にはしない。

### 3.2 旧失敗と補助資料

v3の24/28 wrapper後MemoryError、v4の16/28後phase wall cap、v4 worker terminal欠測は、そのまま履歴として残す。v5の完走で過去の失敗を成功へ変更しない。共通coverageの出力一致は保全・互換性の証拠だが、同じ入力・実装系列の反復を独立した科学的精度証拠として数えない。[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)[S19](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_gpt_review_index_v1.md)

未公開の補助6件は、変更前toy profiler、metadata CLI、保全helperであり、今回の中心的なH4作用・H6案の判断を妨げる不足とは扱わない。必要なsourceが公開済みになったため、追加の一括pushは要求しない。

## 4. H4 v5の検証対象を固定する

対象はlinear H4、隣接距離1.00 Å、STO-3G、legacy DF rank12、8 system qubits、generation-prefix、T=0.8。保存されたnormalized stateを共通targetとする。Nα=Nβ=2のspin sectorは36次元である。small residualだけで基底状態を認定せず、指定stateへのcomplex signalを評価する。[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)[S11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json)

DF Hamiltonianとstateは新たに生成していない。snapshot SHA-256は

```text
3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a
```

である。保存snapshotの生成時Python 3.11.0rc1等のmetadataと、v5実行時Python 3.11.1／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／OpenFermion1.6.1を区別する。[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)[S11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json)

B0はone-bodyとprefixを残しtailをdiscardする。B1は全DFの決定論PF。B2はprefixを決定論的に扱いtailをfinite-RTE化する。B3はprefix0だがone-bodyとscalarを決定論的に残すため、Hamiltonianの全要素を無差別にランダム化した方式ではない。

## 5. 保存された信号結果

次の表は結果報告の丸め値を転記したもの。値は保存signalと保存referenceの差であり、保証付き誤差上界ではない。[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)

| cell | prefix | q | R | K | abs(signal-reference) | log B | 費用測定 |
|---|---:|---:|---:|---:|---:|---:|---|
| B1 S2 | 12 | 1 | — | — | 7.26706e-3 | — | なし |
| B1 S2 | 12 | 4 | — | — | 4.29286e-4 | — | なし |
| B0 S2 discard | 6 | 4 | — | — | 1.88093e-2 | — | あり |
| B2 finite-RTE | 6 | 4 | 8 | 2 | 4.29286e-4 | 3.89982e-5 | あり |
| B2 finite-RTE | 6 | 4 | 8 | 4 | 4.29286e-4 | 3.89982e-5 | なし |
| B3 finite-RTE | 0 | 4 | 8 | 6 | 3.68757e-4 | 8.50757 | あり |
| B1 global S4 | 12 | 1 | — | — | 2.25723e-3 | — | あり |
| B1 global S4 | 12 | 4 | — | — | 9.70940e-6 | — | あり |

**8 correctness cellのすべてについて費用を測ったわけではない。** 現在のB1 S2費用欠測を、旧AX-1bの異なるbuild/control scopeの費用で無条件に埋めない。B2 K4も、K2と丸めsignalが同じだから同じcostと仮定してはならない。

S4 q4がこの表で小さいsignal差を持つことは、強い決定論対照を入れた意義を支持する。しかしq・次数・bias・費用のtrade-offを最適化していないので、S4やPRの資源winnerはまだ決まらない。

## 6. 保存されたwrapper費用とcontrol改善

full measured Hadamard wrapper、state preparationなし、basis `rz,sx,x,cx`、opt1、seed17、backend/coupling指定なしというscopeである。次はcosine axisの値で、sineを含む全記録は[S23](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/wrapper_cost_summary.json)にある。B0/B1は固定回路のn=1、B2/B3は2 trajectoryのsample meanである。[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)

| cell | control | n | RZ count | CX count | circuit size |
|---|---|---:|---:|---:|---:|
| B0 prefix6/q4 | ordinary | 1 | 41,321 | 23,840 | 78,956 |
| B0 prefix6/q4 | symmetric_directional | 1 | 17,753 | 8,800 | 37,180 |
| B2 prefix6/q4/R8/K2 | ordinary | 2 | 44,367.5 | 24,787 | 84,900 |
| B2 prefix6/q4/R8/K2 | symmetric_directional | 2 | 20,851.5 | 9,747 | 43,192 |
| B3 prefix0/q4/R8/K6 | ordinary | 2 | 3,859 | 1,267.5 | 7,371.5 |
| B3 prefix0/q4/R8/K6 | symmetric_directional | 2 | 3,791 | 1,203.5 | 7,239.5 |
| B1 S4 prefix12/q1 | ordinary | 1 | 59,918 | 34,724 | 114,563 |
| B1 S4 prefix12/q1 | symmetric_directional | 1 | 22,514 | 10,994 | 46,865 |
| B1 S4 prefix12/q4 | ordinary | 1 | 238,601 | 138,464 | 456,260 |
| B1 S4 prefix12/q4 | symmetric_directional | 1 | 88,985 | 43,160 | 185,084 |

上の保存値から本レビューで算術計算したRZ削減率 `1-C_symmetric/C_ordinary` は次のとおり。

| cell | RZ削減率 |
|---|---:|
| B0 prefix6/q4 | 約57.04% |
| B2 prefix6/q4/R8/K2 | 約53.00% |
| B3 prefix0/q4/R8/K6 | 約1.76% |
| B1 S4 q1 | 約62.43% |
| B1 S4 q4 | 約62.71% |

この違いは、同じcontrol選択でも、回路構造によって実コンパイル費用への影響が異なることを示す当該sample内の結果である。理論上の任意角rotation会計を、compiler後のRZ数へ一律に1/2と掛ける説明は適切でない。RZには現在のbasisと最適化に依存する構造が含まれ、T-countとも等価ではない。[S06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)[E02](https://arxiv.org/pdf/2511.13855)

B3では決定論backboneがone-body等に限られ、tailは既存ordinary-controlled RTEとして残ることから、対称controlによる削減が小さいことは実装構造と整合的である。ただしgate-category別の因果的費用分解をこのpilotが完了したわけではない。

主解析にsymmetric_directional、ordinaryをpaired感度として残す案は、後続の**新しい比較契約**で採用してよい。旧結果のprimaryを事後変更しない。全方式に適用可能な改善を公平に与え、軸別に有利なpolicyを後から選び分けない。

## 7. Native controlの意味論の独立検討

### 7.1 ordinary controlとdirectional全体を混同しない

[S07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_control_plan.py)はforward halfをUNCONTROLLED、逆順halfをDIRECTIONAL、中央tailをORDINARYにする。名前にdirectionalが含まれていても、最終的に狙う作用は`diag(I,U)`であり、全体を`diag(U†,U)`へ変える設計ではない。[S06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)[S07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_control_plan.py)

以下はsourceに対応する本レビューの代数的確認である。回路の時間順に前半へE1,…,Emを置き、operatorとしての積をF=Em…E1とする。同じ符号の逆順後半をS=E1…Em、中央のsampled unitaryをVとすれば、control=1でS V Fとなる。control=0では中央がI、後半がE1†…Em†=F†となるため、F† I F=Iである。従って理想的なunitary elementary gateの意味論では、全stepは`diag(I,S V F)`になる。q回の積と、別のcontrolled scalar phaseを含めてもordinary controlの形を保つ。

この確認は状態ψに固有の近似ではなく、理想gateの代数である。一方、実装のfloating-point誤差、basis分解の精度、実際の全branchに対する数値検査範囲は別である。H4の有限probeの一致を、全Hilbert空間の数値的認証と呼ばない。

Simon–Loveの一次資料も対称PFで一部をdirectional化してordinary-controlled evolutionを構成する考えを扱っている。現在のDF-native接続をcontrol原理そのものの新規発明とはしない。[E02](https://arxiv.org/pdf/2511.13855)

### 7.2 scalar phaseとtailの扱い

`append_directional_diagonal()`はRZ/RZZの符号をancillaで反転し、global phaseに由来するancilla回転を保持する。分子constantとextracted identityは最後のcontrolled phaseへ反映される。Hadamard信号では相対位相なので、global phaseを単に無視したunitary一致検査では不十分である。[S06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)

中央tailはexplicitなunitary event列である。finite平均の非unitary多項式をgateとしてcompileしていない点は適切である。原tail builderと同じrequest/event列を再利用し、その順序・phase・identityを維持する設計を支持する。

### 7.3 S4の具体的な公式

現在のglobal S4は、3つのS2を合成するYoshida型である。

\[
S_4(t)=S_2(wt)S_2((1-2w)t)S_2(wt),\qquad
w=\frac{1}{2-2^{1/3}}\approx1.35120719196.
\]

中央係数は約−1.70241438392である。単に「標準4次」とだけ記して、5-piece再帰型のstage数や費用を流用しない。globalなDF term列の四次化であり、partial方式のinner backboneだけを四次化したものではない。[S07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_control_plan.py)[S08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)[S09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py)

当該S4の未融合half-stageは、H4 q1で約+0.540482877、−0.680965754、q4で約+0.135120719、−0.170241438を含む（T=.8からの算術）。primitiveの独立全列検査±.2だけではこれらを覆わない。merged PF iteratorはさらに融合した時間を持ち得るため、両表現の実時間集合を展開して確認する必要がある。

ただし、現在の終端state／signal比較は登録cellの実時間を通して実行されている。従って「四次の実時間では何も検査していない」という評価も誤りである。欠けているのは、実時間集合に対応する局所誤差と、その累積の説明である。

## 8. State-action、sector、参照計算の強みと限界

### 8.1 何が改善したか

有限Taylor numeratorはHorner再帰でvectorへ作用し、非unitary中間状態を再正規化しない。raw経路では各micro-stepでb_Kによって割り、corrected経路ではそのまま多項式を反復する。rawとcorrectedを別の数値経路で計算する設計は、単にrawへBを掛けるだけよりも、conditioningの区別を明確にできる。[S08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)

sector構成について、各dΓ(g)のcross-spin係数がexact zeroであること、basisの完全性と順序、norm、OpenFermionとQiskitのbit orderを検査している。full-vectorで完全primitiveを適用してからleakageを確認・投影し、Gaussian gate一つごとに中間状態をspin sectorへ切り詰めない点は重要である。[S05](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_h4_science_v5.py)[S08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)[S11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json)

### 8.2 共通する誤りをどこまで排除できるか

`dense_df_qiskit()`とsector matvecは、どちらも既存`df_linear_operator()`を経由する。full/sector比較はsector・basis・投影の一致に有用だが、共通matvec定義の系統誤りを完全に排除する独立oracleではない。同じdense matrixへのexpmとeighも、入力構築の正しさを独立に保証しない。[S05](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_h4_science_v5.py)[S17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/df_hamiltonian.py)

一方、native circuitとspectral primitiveの比較は実装経路が異なり、occupation energyを使う小型toyの全unitary比較もある。これらを総合して実装整合性を評価すべきで、同じコード由来だから無価値とする必要もない。[S09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py)

次のH4限定検証では、保存された同じDF係数から、独立したoccupation-basis構築または別の小型構成を一つ用意する価値がある。高精度化だけでは共通の誤ったHamiltonianを直せないため、**構成の独立性**と**演算精度の独立性**を分けて計画する。

### 8.3 残差は何を保証するか

正規化state ψ、Hermitian H、E=<ψ|H|ψ>、ρ=||(H−E)ψ||について、厳密算術では

\[
\|e^{-iHT}\psi-e^{-iET}\psi\|\le |T|\rho
\]

であり、対応するsignal差も同じ上界で抑えられる。Duhamel型の積分表現から得られ、ψが基底状態であることは要しない。

保存記録では|T|ρ≈7.53918e−16に対し、計算されたphase surrogateとreference signalの差は約1.39111e−15である。[S11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json) これは理論の反例ではない。数値的に求めた残差、energy、exponential、inner product、state正規化の丸め誤差が上式へ自動的には含まれないことを示す具体的な注意例である。

小さい残差は近似固有stateの診断であり、最低固有値に対応することの認定ではない。現研究のprimaryは指定stateへのsignalなので、毎回厳密なground-state証明を要求する必要はない。

## 9. H4の新しい科学的情報を評価する

### 9.1 B0：pure discard/PF分解のデータ不足は一部解消した

B0の一次記録にはexact truncated-H signalがあり、

\[
z_{\rm B0,PF}-z_{\rm ref}
=(z_{\rm trunc,exact}-z_{\rm ref})
 +(z_{\rm B0,PF}-z_{\rm trunc,exact})
\]

が複素量で保存されている。[S12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B0_q4_correctness.json)

保存値から本レビューで計算したノルムは、discard差約0.0192240851、PF差約0.0004148149、合成差約0.0188093141である。両成分は概ね反対向きで一部相殺している。これは当該p6/q4でdiscard成分が主要であるという解釈を支持する。qを増やすだけでdiscard誤差が消えるとは期待できないが、全prefixやB0全体の非有用性は導かれない。

旧AX-1の段階で「exact truncated-H signalがなく分解不能」とされた点は、**現在のH4 p6/q4については保存値が追加された**。この進展を反映する一方、全旧candidateの分解が揃ったとはしない。

### 9.2 B2：登録点ではfinite cutoffよりouter-PF側の差が大きい

B2 K2の保存finite差のノルムは約6.72e−12、total差は約4.29e−4である。K4も報告上同じ丸めtotalとなっている。[S13](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B2_K2_correctness.json)[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md) 当該記録は、Kを増やすだけでは支配的なtotal差を大幅に下げない可能性を示す。

ただし、1e−12程度の差そのものは未認定の数値誤差と比較する必要があり、「真のfinite-RTE誤差が厳密に6.72e−12」とは表現しない。totalの差分構造を調べる材料として用いる。

### 9.3 B3：小さいbias、短い回路、大きいnormalizationは両立する

B3では保存total差約3.69e−4、symmetricのsample RZ約3,791である一方、B=4952.101276245117である。[S14](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B3_K6_correctness.json)[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)

保存Bを平方しただけの本レビューの算術では、

\[
B^2\approx 2.4523307\times10^7.
\]

従って、同じheadroomのHoeffding会計ではnormalizationだけで大きな測定負担となり得る。ここでは新しいN/Gを確定計算したり、B3が方式全体として必ず負けると認定したりしない。r/R/partitionを変えた場合のBとcostの競合は未探索である。

signed finite差は約7.11e−13に留まる。この記録は、canonical LCU normalizationの大きさと、指定stateでのfinite平均の近似biasを同じものとして扱えないことを示す。なぜ小さいfinite差になったかをstate/spectrum構造から確定したわけではない。

### 9.4 global S4：強い対照を含める研究上の意味

S4 q4の保存real/imag差は約−9.58e−6／−1.57e−6、RZ sampleは88,985（symmetric）である。[S15](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B1_S4_q4_correctness.json)[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md) より小さいbiasのために1-shot costを増やす対照が実装可能になった点が重要である。PRとの比較は、同じqのgate数だけでなく、同じ精度条件のN×Cで行う必要がある。

現在のS4は特定公式であり、あらゆる高次PFの最良対照ではない。後続の広い優位claimでは比較範囲を明示する。ただし、今回の技術pilotに全高次公式を追加することは求めない。

## 10. 最重要の判断：数値誤差をどの強さで扱うか

### 10.1 三つの証拠水準を分ける

| 水準 | 必要な根拠 | 言えること |
|---|---|---|
| Technical agreement | 数値gate、異なる経路の一致、norm/leakage/phase検査 | 登録実装の整合性、bounded feasibility |
| Empirical numerical validation | 独立構成・精度/経路比較・中間norm・headroomへの感度、明示した数値仮定 | 条件付きの精度・資源研究、頑健性の観測 |
| Certified numerical bound | 誤差を含む上界、保証付き演算または妥当な解析、全必要経路の会計 | 条件を満たす範囲で保証付き適格性・十分shot会計 |

**本研究を続けるために、全gate・全state・全cellについて形式的な区間演算認証を完成することを必須にしない。** それは別の大規模研究へ主題を移し得る。

一方、経験的差を`u_bound`と命名することも認めない。現段階は、最小限の独立数値検証で`u_empirical`の根拠と適用範囲を整え、下流の結論をその数値仮定への感度付きとして報告する方針を採る。厳密なcertificateが得られた成分だけを別に記録する。旧v5の`numerical_allowance_certified=false`は変更しない。

### 10.2 targetを明確に定義する

保存binary64係数が表すH_DFと、保存vectorを正規化した指定ψを理想的な数学的targetとして定義する案が扱いやすい。原の分子積分Hamiltonian、無限精度のDF分解、真の基底状態へ黙ってtargetを置換しない。

正規化前後の差、Hermitization前後の差、DF truncation、係数cutoff、one-body補正をそれぞれ追跡する。targetを保存後H_DFとする場合、元積分Hとの差は表現誤差層であり、PF signal biasへ二重に加算しない。[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)[S17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/df_hamiltonian.py)

### 10.3 uに混ぜるべきでないもの

PF、finite cutoff、discardは方式のモデルbiasである。参照exponential、state-action、normalization、inner productの浮動小数点誤差とは分ける。モデルbiasが大きいことを「数値誤差が大きい」と表現せず、丸め誤差が未評価であることを「PRが不正確」と読み替えない。

## 11. 非unitary平均の誤差伝播とheadroom

### 11.1 作用ごとの誤差伝播

理想stage A_j、計算stageの局所誤差η_j、入力誤差e_jに対し、妥当な上界L_j≥||A_j||があれば、

\[
e_{j+1}\le L_j e_j+\eta_j,
\quad
e_s\le e_0\prod_{j=0}^{s-1}L_j+
\sum_{j=0}^{s-1}\eta_j\prod_{k=j+1}^{s-1}L_k.
\]

この式は、本レビューで誤差会計の設計を説明するために示した一般的な導出である。実際に全η_j/L_jを評価したわけではない。

決定論的な理想unitary stageのnormは1だが、finite polynomial stageのnormは1とは限らない。`finite_taylor_action()`が中間vectorを正規化しないのは、そのため正しい。[S08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)

raw signalの誤差をcorrectedへ移す場合はBによる増幅が必要である。一方、直接corrected多項式を評価する経路では、その経路自体のconditioningを調べる。**Bが大きいことだけから、corrected経路の誤差が必ず同じB倍であるとはしない。** Hermitian tailのspectral範囲で多項式のnormを評価するなど、粗いBの積より適切な評価が可能かを検討する。数値で求めたnormを無条件に厳密上界とはしない。

### 11.2 三状態適格性

保存近似signalをẑ_x、参照近似をẑ_refとし、

\[
\widehat b_{x,a}=|\operatorname{axis}_a(ẑ_x-ẑ_{\rm ref})|,
\quad b_{x,a}\in[\max(0,\widehat b_{x,a}-u_{x,a}),\widehat b_{x,a}+u_{x,a}].
\]

上の区間が保証付きであるのはuが保証付きの場合に限る。empirical uでは条件付き診断である。

対称軸配分ε_a=ε_sig/√2について、両軸の上側biasがε_aより小さい場合にその規則上のELIGIBLE、どちらかの下側biasがε_a以上ならその規則上のINELIGIBLE、その他はUNDETERMINEDとする。**対称軸配分規則で不適格であることを、あらゆる推定器でtask不可能と同一視しない。**

保証付き会計をするなら、

\[
h_{x,a}=\epsilon_a-\widehat b_{x,a}-u_{x,a}>0,
\qquad
N_{x,a}=\left\lceil\frac{2(B_x^{\rm upper})^2\log(2/\alpha_a)}{h_{x,a}^2}\right\rceil
\]

を使える。Bの評価誤差も含める。原案のα_real=α_imag=.025はper-candidateの失敗確率配分であり、全candidateの統計的winnerを同時認定する保証ではない。

丸めを除けば `d log N / d b = 2/h` である。従って、uを単にεの1%以下にするだけでは境界付近で十分でない。uと残りの誤差予算hの比、数値水準を変えたときの適格性・順位の安定性を確認する。

### 11.3 実装選択への注意

高精度演算、別の行列構成、matrix exponential actionの比較など具体的なbackend選定はCodexに委ねる。SciPy 1.14.1の`expm_multiply`には公開API上の`tol`引数がないため、「tolを下げた」とだけ記す再現不能な検証案を作らない。実際の精度水準・別経路・内部設定を明記する。[E04](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.sparse.linalg.expm_multiply.html)

## 12. Finite平均とsampled estimatorの接続

[S08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)と[S05](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_h4_science_v5.py)は、corrected多項式とraw多項式/bの整合性を検査している。[S09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py)はexplicit event列のnative/legacy replay一致やwrapperのtoy期待値を検査している。しかし、それらをもって全finite分布の確率重み付き平均とmolecular wrapperが結合済みとみなしてはならない。

必要な接続は概念的に、各micro-stepで

\[
\mathbb E[V_\omega]=P_{K+1}(-i\tau\overline H_R)/b_K(\tau)
\]

がphaseを含めて成立し、各occurrenceのdrawが契約どおり独立で、ordered productの平均が対応するordered平均作用の積になることである。各V_ωはunitary event列であり、Hadamard出力の平均がその実部・虚部を測る。[S06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)[S08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)[S09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py)

追加すべき最小検証は、既存sampler／finite平均helperを利用した小型fixtureの完全列挙による確率重み付き比較、代表的なmolecular prepared eventのphase・basis・register対応の検査、micro-step／occurrence／shot間のfresh IID規則の確認である。全部のH4 trajectoryを巨大列挙したり、n=2を大量のsignal Monte Carloへ置き換えたりする必要はない。

同じexplicit trajectoryをordinary/directionalと両axisの**費用測定**に共有することは、paired比較として合理的である。しかし量子測定では各shotにfresh trajectoryを生成するという別の意味論がある。2本の固定回路を大量に繰り返す実験を、fresh IID平均の代替として扱わない。

この接続が未解決なら、決定論的なfinite多項式のsignal数値は検査できても、N_aの測定会計をそのまま実装へ結び付けたとは主張できない。

## 13. H6入力policyに見つかった具体的な問題

### 13.1 `df_rank=None`はbuilderの層によって意味が違う

H6案は`df_tol=1e-8`、`final_rank`指定なし、actual rankを保存する方針である。[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)[S16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json)

しかし現行`build_df_h_d_from_molecule()`は、渡されたdf_rankをまず

```python
resolved_df_rank = resolve_df_rank_for_molecule(molecule_type, df_rank)
```

で解決する。`resolve_df_rank_for_molecule()`はdf_rankがNoneなら分子別configへ戻り、H6ではselected_rank=11、H8では15を返す。続く`_low_rank_kwargs()`はresolved rankを`final_rank`として、df_tolを`truncation_threshold`として両方渡す。[S17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/df_hamiltonian.py)[S18](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/config.py)

OpenFermion v1.6.1は`final_rank`が指定されていればthresholdからrankを選ばず、そのrankを使う。[E03](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py#L76-L169) したがって、既存分子builderを素朴に

```python
build_df_h_d_from_molecule(6, df_rank=None, df_tol=1e-8)
```

と呼ぶと、H6案の「tolだけでrankを決める」という政策と一致しない。

**これは将来のH6入力portで解消すべき実装・契約上の不一致であり、まだ生成していないH6が実際に誤っていたという報告ではない。** 既存H4は明示rank12の保存targetであり、この発見を理由にその結果を無効にしない。

### 13.2 修正の科学的条件

tol-onlyを採るなら、configによるrank補完を明示的に無効化した新しいadapterまたは直接integral-to-DF経路を使う。既存の分子別configを全研究に影響する形で書き換えない。

次の情報を保存・検査する：入力policy、要求tol、実際にdecompositionへ渡したkwargs、final_rank不指定、actual L、返却truncation value、係数ordering、Hermitization差、one-body補正、後続cutoff。synthetic/mock検査でNoneから11へ戻らないことを先に確認する。

`df_tol=1e-8`はH6技術pilotの具体案として維持してよい。ただし値を採用したことだけで原化学Hamiltonianへの誤差保証が得られたとしない。主資源比較のH4/H6/H8系列では共通policyを別途固定し、legacy H4との同等性がなければ接続点を必要とする。

### 13.3 DF truncation valueと表現誤差

OpenFermion v1.6.1の実装は、`|lambda_l| (sum_pq |g_lpq|)^2`の重みを作り、降順に並べた後、total cumulative sumからprefix cumulative sumを引いてdiscarded-tailの会計を返す。APIの短い説明だけでなく実装を参照する。[E03](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py#L76-L169)

厳密な係数と適切な演算規約の下では、`||a†_p a_q||<=1`より、discarded blockの作用素ノルムはそのような重み和で保守的に抑えられる。ただし、分解のroundoff、Hermitization、別のcutoff、積分規約・one-body correctionの誤りを自動的に含むわけではない。

現repositoryの`_as_hermitian()`は対称化を返し、skew normが指定tolを超えていてもその場で例外を出さない実装である。[S17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/df_hamiltonian.py) 新入力については、対称化前後の差を記録し、計画外の大きな変更を無検査で許容しない。これも、既存H4で大きな差が発生したと確認したという意味ではない。

## 14. 不足検証11項目の優先順位

原案のIDを維持し、すべてを同時にH6技術pilotのblocking条件にするのではなく、必要になる判断点を分ける。[S02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_validation_gaps_v1.md)

| ID | 独立レビューの優先順位 | H6 technical前の最低条件 | 本格資源比較までの条件 |
|---|---|---|---|
| U-N1 target/state/u | 最優先 | 共通targetとu証拠区分を明文化、未知は未知と記録 | target・state・uの一貫した会計 |
| U-N2 reference roundoff | H4限定検証を優先 | 参照経路・検査計画・technical scopeを固定 | 独立構成/精度比較、u_empiricalまたはu_boundの根拠 |
| U-N3 stage時間/非unitary | 最優先 | actual time集合、norm/overflow、計数上限を定義 | 累積誤差とheadroomの説明 |
| U-N4 u-aware shots | 早期に実装 | 純technicalでN/G=nullなら未使用可 | 三状態・log-domain・u=0旧再現・数値根拠 |
| U-N5 sampled estimator接続 | 科学的会計への必須条件 | 小型の接続検証と明確な未検証範囲 | finite平均とfresh-shot推定量の接続 |
| U-C1 main探索 | pilotでは限定契約で可 | 7 cell/36 wrapper、policy/seed/count固定 | 共通prefix、q/R/K、強い対照、boundary・quota |
| U-C2 cost母平均 | pilotはn=2を維持可 | 工程sampleと表示、winnerを出さない | 標本設計・rare event・独立confirmation |
| U-H1 H6 input/DF | H6前に必須 | tol-onlyとconfigを分離、input/hash/sector/state固定 | 共通DF政策とlegacy bridge |
| U-H2 orchestration/caps | H6前に必須 | 別controller、before-call cap、watchdog、source固定 | main予算・partial/失敗の一貫した記録 |
| U-P1 predictor契約 | 支援課題 | 旧FEW凍結、order/control外はN/A、truthを混ぜない | サイズ対応特徴と予測時点・取得費用 |
| U-I1 H8独立性 | H8前に必須 | H6全露出をdevelopment扱い、H8へ接触しない | H8前のモデル/仮説/候補/標本freeze |

U-N2/U-N3の厳密bound完成を、あらゆるH6工程測定の前提にはしない。一方、意味論やtargetが不明な状態で技術pilotを進めてよいという意味でもない。何を検査し、何を判定しないかを明示する。

## 15. H6 7 cell・36 wrapper案の評価

### 15.1 候補構成は技術目的に対して合理的

p=(L+1)//2、Lは実際のDF rankとする。[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)[S16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json)

| cell | order/方式 | prefix | q | R | r | K | replica |
|---|---|---|---:|---:|---:|---:|---:|
| H6_B0_S2_q2 | 二次discard | p | 2 | — | — | — | 1 |
| H6_B1_S2_q1 | global S2 | L | 1 | — | — | — | 1 |
| H6_B1_S2_q2 | global S2 | L | 2 | — | — | — | 1 |
| H6_B1_S4_q1 | global S4 | L | 1 | — | — | — | 1 |
| H6_B1_S4_q2 | global S4 | L | 2 | — | — | — | 1 |
| H6_B2_K2_q2_R4 | partial S2 finite-RTE | p | 2 | 4 | 2 | 2 | 2 |
| H6_B3_K6_q2_R4 | partial S2 finite-RTE | 0 | 2 | 4 | 2 | 6 | 2 |

36 wrapperは、deterministic 5×2 controls×2 axes + random 2 cells×2 replicas×2 controls×2 axesである。4 random trajectories、8 outer occurrencesとなり、machine-readable案の数と整合している。[S16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json)

B0を含め、S2とS4の両方をcost測定する点は、H4で不足した比較coverageを技術的に補う。K2/K6はfinite経路の異なるcutoffを通すための代表であり、K最適化の証拠にはならない。ここへ多数のprefix、K4、q8/R64を追加しない。

### 15.2 小さいRは科学的に容易な条件とは限らない

τ=λ_R T/Rなので、同じλ_RならRを8から4へ減らすとτは2倍になる。H6ではλ_R自体も未知である。q1/q2・R4で回路回数を抑えても、normalization、非unitary中間norm、raw underflow、finite biasが改善するとは限らない。

したがって、actual inputからλ_R/log B等の定義可能性を事前に確認し、非finite、scaleに合わないcomparison、raw underflow等の停止規則を結果前に固定する。発散や不適格が出た後にR・rank・tol・seedを変えて同じpilot成功へ救済しない。

### 15.3 referenceとstate

400次元sectorの単一dense reference matrixと、4096次元full-vector native actionを併用する案を支持する。全4096×4096のfragment matrix/eigenvector cacheを作る必要はない。各complete primitiveのsector保存性を確認し、途中のGaussian操作はfull vector内で扱う。

eigshのHF開始vector、which=SA、tol1e−12等は技術案として妥当だが、収束flagと小残差だけでground-state証明にしない。指定stateへのreference signalが主である。exact cross-spin zerosでspin sectorが成立しない場合は、粒子数sectorへ無断で切り替えず別仕様へ戻す。

### 15.4 実行前に修正・確定する事項

最優先は第13節のtol-only入力経路。併せて、uの証拠区分、全stage時間、finite平均／sample接続、actualrankでのtask展開、source/input hash、source scopeを満たすsynthetic testsを固定する。

全時間×全sector列を機械的に増やしてprimitive上限2,000に収まらなくなった場合、上限に合わせて検査を黙って削除しない。構造的なsector証明、独立small-reference、代表的な数値検査の役割を整理し、必要なcoverageと予算を結果前に決める。

H6技術pilotでは数値allowanceが未認定でも、明示したtechnical契約とguardの下で作用・負荷の検査をしてよい。ただし、`numerical_allowance_certified=false`、`accuracy_eligibility=UNDETERMINED`、N/G=nullを維持する。これは本レビューが採る進行方針であり、現状の未実装launcherへの起動認可ではない。

## 16. H4で追加すべき最小検証

### H4-N：入力・参照と数値水準

保存H4 target/stateを変えずに、独立した小型sector構成と異なる演算精度または別のreference経路を照合する。生成・basis・scalar・one-body correctionが一致しているかを先に確認する。expm/eighの差に任意の安全係数を掛けただけのuを厳密上界にしない。

### H4-A：8登録signalの実時間・有限平均

既存8 cellのsignalを対象として、実際のprimitive time集合、Horner／spectralの独立性、中間norm、raw/corrected、誤差の累積とinner productまでを記録する。finite差がroundoff水準にあるcellは、その差の桁を物理的に解釈しない。

### H4-E：sampled estimatorへの接続

小型のfinite event列挙、phaseを含む平均operator、ordinary/symmetric／cosine/sineを繋ぐ検証を追加し、molecular scopeでは同じprepared event semanticsが使われることを確認する。ランダム回路を大量samplingして平均signalを近似し直す計画を必須にしない。

### H4-M：u-aware会計の純synthetic検証

u=0で旧式との整数一致、境界、片軸だけ未確定、largeB、overflow、B上側値の扱いを検証する。実際のN/Gを出す場合は、そのu水準を明示した新しい解析として別に記録する。

**これらを理由に、完了済み28 wrapperを一律再compileする必要はない。** 既存sourceが正しく対応する範囲の保存costを保全し、semantic不一致やcompile後作用に関する具体的な懸念が見つかった場合だけ、必要なtargeted確認を別scopeで計画する。

H4のS2費用など不足する比較cellを追加する必要性は、本格資源比較の契約で決める。数値uの問題を解くための必須作業とは分離する。

## 17. 資源計画と実行監査

H4の実測はparent wall787.984819秒、peak RSS約3.290GiBである。numeric fingerprintのinclusive wall622.620秒はparent wallの約79%に相当するが、他stageとnestedなので足し合わせない。これを削除した場合のspeedupやH6の所要時間を保証しない。[S04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)

H6案の8GiB AS、total7,200秒、36compileは、まだ未割当のproposalである。入力生成wall budgetはJSONでもnullであり、pilot側input/reference phase1,800秒と入力生成を混同しない。[S16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json)

| H6案の項目 | 評価 |
|---|---|
| CPU/BLAS/worker各1 | 初回profileとして合理的。実割当を確認 |
| AS8GiB | 仮上限。RSSと別。H4完走から保証しない |
| phase1,800/1,800/3,600秒、total7,200秒 | 合計整合。import・起動overheadもtotalに含める |
| output512MiB/log64KiB/diag1,024 | partial/terminal用余裕を含めてbefore-write enforce |
| solver/reference matvec cap | counterを読むだけでなく呼出し前に止める |
| untranspiled/transpiled instruction cap | 展開・hash・compiler memoryの上界ではない |
| retry/resumeなし | 今回の登録pilotでは維持。失敗を残し別判断へ |

単一complex128配列の算術サイズでは、H6のfull vector64KiB、sector matrix約2.44MiBに対し、full dense matrix1枚256MiBである。fragment行列と固有vectorを多数cacheすると急に大きくなる。H8全空間dense64GiBは使用しない。小さいstatevectorだけを見て回路DAG、基底分解、fingerprintのmemoryを過小評価しない。[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)

fingerprintを主要な古典費用として観測したことはengineering上の知見だが、その最適化をTrack Aの主研究課題へ置き換えない。call-local cacheと群単位解放等の既存対策を保ちながら、H6で実際の負荷を測る。

今後のsourceは、可能なら実行前にsource commitを作り、plan・authorizationをそのsourceへ結び付ける。sourceと認可を同じcommitへ自己参照させず、実行時HEAD一致gateを破らない設計にする。過去のlocal-byte executionを現在の正式commit executionへ書き換えない。

## 18. 期待費用と本比較の統計

n=2は工程profileとしては利用できるが、random cost母平均の精度やrare eventの寄与を認定するには不足する。SD=0やordinary/directionalの差が大きいことを、全candidateの統計的順位保証へ拡張しない。[S01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_post_scientific_review_v1.md)[S02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_validation_gaps_v1.md)

本比較では、同じtrajectoryのaxis/control pairingを保ち、必要なcovarianceを残す。候補を選んだ費用標本だけでその候補を最良と確定しないよう、主要な境界・winner候補について独立confirmationや事前登録した精密化を検討する。古典cost sample数と量子shot数Nを別fieldにする。

prefixとq/R/Kの候補範囲は、方式ごとに適用可能な自由度を公平に与える。p≈L/2の一つのB2だけでPR方式の最適性能を代表しない。B2 K2とB3 K6の比較にはcutoffもpartitionも違うため、観測差をランダム化割合だけの因果効果としない。

主指標は登録したlogical RZ-workであり、任意角rotation、non-Clifford cost、T/Toffoli、実機runtimeは別である。state preparationを除く比較はそのscopeで記述し、必要なら共通準備費用Pに対する感度`G_x(P)=G_x(0)+N_total,x P`を別途使う。本pilotでPや物理資源を推定したわけではない。

## 19. 研究としての価値・新規性・着地点

PR原論文はsingle-ancilla phase estimationの化学系benchmarkに対する詳細な資源見積もりを既に示している。したがって、PRを化学系へ適用したことや資源を数えたこと自体を初の貢献としない。[E01](https://arxiv.org/abs/2503.05647v2)

また、対称PFのcontrol改善は既知であり、今回の実装はその原理をDF-nativeのphase・tail・wrapperへ正しく接続した技術的成果である。[E02](https://arxiv.org/pdf/2511.13855)[S06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)[S07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_control_plan.py)

今回のH4 v5だけで独立論文の主要な科学的結論を完成したとは判断しない。一方、研究としては、強い決定論対照とdiscardを揃え、回路費用・normalization・bias margin・DF政策のそれぞれが競争力へ与える影響を説明する方向に進展している。

**中心RQ-R**：同じDF target・state・finite-time signal・測定規則の下で、PRはどの精度・partition・サイズ・実装条件で資源的に競争的か。非優位の場合、その原因をどの成分で説明できるか。

**補助RQ-P1**：その比較を支援する費用モデルにはどの構造情報が必要か。旧FEWはorder/control/サイズの適用範囲を外れる場合N/Aまたは外挿診断とし、今回の少数costで再fitしない。

operational bias predictorなしの総資源予測RQ-P2は未達成として分離する。これを完成させるまで直接resource studyを止める必要はない。

本格的な目標は、H6 developmentで整理した資源差の説明や仮説を、H8に接触する前に固定して独立確認することである。単にH4/H6/H8を3点並べて漸近scalingや化学一般の優位を主張しない。

今回の外部確認はPRとcontrolの中心一次資料、および使用版ライブラリの意味論に限定した。関連論文全体の網羅的再検索を完了したとはしておらず、「新規性確定」「投稿先で採択可能」といった保証は出さない。以前の文献対応・旧paper_d6の版問題は、未解決のまま保持する。[S27](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax0_model_correspondence.md)

## 20. 代替進行案の比較

| 案 | 利点 | 問題 | 採否 |
|---|---|---|---|
| 直ちにH6を無修正launch | 速く次サイズへ行ける | DF rank政策の不一致、未seal入力/caps、意味論不足を持ち越す | 不採用 |
| 全数値演算の厳密certificate完成まで全研究停止 | 強い保証を目指せる | 必要以上に主題を数値認証へ移し、technical feasibilityも得られない | 不採用 |
| H4全28wrapper再compile＋大幅sampling増 | 費用統計を増やせる | reference/u/finite平均接続の主問題を直接解かない | 現段階では不採用 |
| **H4の限定検証＋H6技術準備を一つの作業束へ** | 重要な不確かさと実装policyを閉じ、次サイズへの道筋が明確 | 個々の実行scopeは明示が必要 | **採用** |
| H4 FEWをさらにfit | 既存データで進めやすい | 現在のblockerとRQ-Rに直結しない、事後最適化に偏る | 不採用 |
| H4を閉じて直ちにQPE/物理資源へ拡張 | 応用目的へ近い | task・合成・state preparationまで広がり、現検証不足を解消しない | 今回は分離 |

この判断は「PRが必ず有利になる」ことを前提にしない。強いbaselineが勝つ場合も、比較の公平性と原因分解が成立すれば正常な科学結果である。

## 21. 次にCodexへ渡す作業範囲

次は**Codex**が担当する。以下を一つの準備・検証パッケージとしてまとめる。

### 作業束A：H4の数値・意味論を閉じる準備

第16節のH4-N/A/E/Mを具体化し、既存snapshot・source・旧結果を保全する。数値検証のbackend、独立構成、精度水準、stage集合、記録schema、test内容はCodexが設計する。新しい分子計算・既存H4の追加数値実行は、対象と予算を明記した後続の明示的実行指示に従う。

### 作業束B：H6 draftと実装準備

7 cell/36 wrapperの技術目的を維持し、tol-only入力経路を明示してconfig fallbackを防ぐ。H6専用controller、bounded reference/solver、before-call caps、source/input binding、失敗recordを作る。H4 controllerの8qubit/rank12/256固定を文字列置換で緩めるだけの移植はしない。

### 作業束C：実行準備の確認と停止

source・synthetic tests・plan・未確定input/資源をまとめて保存する。実行前に必要なsource/results/auditがremoteから読めるよう、対象だけをcommit/pushしhash対応を残す。機密情報、dirty差分、Track B、旧契約・結果は保護する。

本レビューから、H6入力作成やH6科学runnerの起動を自動開始する権限は発生しない。準備が整ったら、ユーザーの明示的な実行指示にscopeと上限を載せる。科学的条件が変わらない通常の実装・synthetic修正・資料整備のたびに、新たな本格研究レビューを要求しない。

**早期にGPTへ戻る条件**は、state/target・主要metric・PF/control意味論・独立性・主たる比較範囲を変更する必要が判明した場合、または正しさを左右する未解決矛盾が出た場合である。単なるpath、logging、data schemaなどの保全的修正はCodexの範囲でよい。

## 22. 次の科学的GO/STOPと研究の終了条件

### H6 technicalへの進行条件

入力policyが実装と一致し、target/state/sector/sourceが追跡可能で、登録7cellの作用と数値scopeが定義され、hard cap・監査・失敗処理・別launch認可が揃うこと。精度認定をしないtechnical contractなら、総uの厳密認定は起動前必須にしない。

### H6本検証への進行条件

H4限定検証とH6 technical結果を受け、少なくとも数値検証の根拠とheadroom、finite平均／sampled estimator接続、公平なmain候補・baseline・cost統計、計算予算をGPTが評価する。technical完走のみでは本検証へ自動進行しない。

### H8への進行条件

H6までのモデル・仮説・候補生成・標本・主張をfreezeし、H8で何を独立に識別するのかと予算を確認する。H6で修正した内容をH6自身の未使用検証とは呼ばない。H8への先行接触はしない。

### 研究を縮小または止める条件

意味論の不整合を解消できない、数値不確かさが主要な判断を不安定にする、main比較のfairnessが確保できない、必要な計算が予算に入らない、既存研究との差分が新しい知見にならない場合はscopeを縮める。PR非優位、モデル一致、強い決定論法が勝つこと自体は停止理由にしない。

## 23. 3文書への具体的な反映事項

| 対象案 | 維持する内容 | 反映すべき変更 |
|---|---|---|
| H4後科学レビュー案[S01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_post_scientific_review_v1.md) | 技術完了と資源認定の分離、RQ-R主、phaseとuの区別 | 厳密uを全研究の一律前提にしない。経験的数値検証と保証を別層に固定 |
| 不足検証一覧[S02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_validation_gaps_v1.md) | 11IDと再利用source | 第14節の判断点別優先順位、B0分解の保存済み範囲、tol/config問題をU-H1へ追加 |
| H6契約案[S03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)[S16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json) | 7cell/36wrapper、CPU主体、development扱い、N/G未認定 | tol-only adapter、actualkwargs監査、actual time・scale/overflow、入力生成別予算、未認定uでのtechnical scopeを明記 |

旧文書を黙って「当初から承認済み」へ変えず、新しいamendmentとして本レビューとの対応を残す。既存の結果・source freeze・失敗履歴は変更しない。

## 24. まとめ

H4 v5は、native DF、global S4、対称control、finite平均、bounded実行が登録条件で接続できたことを示した。B0のsigned誤差分解、B3のlarge normalization、control改善の非一様性は、RQ-Rを進めるための具体的な材料である。

一方、8/8 correctnessと28/28wrapperだけで精度認定や測定込み資源winnerを結論できない。次に解くべき問題は、H4の数値検証とestimator接続、H6のinput policy・capsである。**特に`df_rank=None`のconfig fallbackは、H6生成前に修正しなければならない。**

本レビューは、次にCodexへ渡す研究方針と作業境界について完了した。以後は同じ問いを毎commitで再レビューせず、H4限定検証とH6 technicalを通じて新しい科学的判断が必要になった節目でGPTへ戻す。H6の実行認可、H6本検証、H8独立評価は別であり、現在の未認可状態を維持する。

---

## 付録A：本レビューで行った算術と行っていない計算

保存値だけに対する算術：B3のB²=24,523,307.05018852、5 cellのcontrol別RZ削減率、B0/B2/B3の保存複素差のノルム、Yoshida係数からのhalf-stage時間、inclusive fingerprint wallとparent wallの比。

これらは新しいmolecular signal、Qiskit compile、random sampling、fit、uの認定、N/Gの確定計算ではない。特にB3のB²を示しただけで総資源順位を認定していない。

独立に再実行していないもの：167 tests、H4 v5 runner、旧v3/v4、保存verifier、全163sourceのbyte照合、全447raw fileのhash再計算。実施済みとの記述は保存監査の内容として引用した。新たに得た実験結果として扱わない。

## 付録B：判断変更の履歴と根拠

AX-1b後の「RQ-R主／RQ-P1補助」は維持した。今回、H4技術pilotが完了したため、単なる実装準備から数値・estimator接続とH6への段階へ進む判断を行った。

新しく明確にしたのは、(a)技術pilotと厳密な資源certificateの必要条件を分けること、(b)H6 tol-only入力で既存config rankを無意識に使わないこと、(c)全11項目を同時に閉じる必要はなく判断点別に優先すること、(d)H4全wrapper再compileより限定数値検証を優先することである。

これは新たなsource読解と保存結果に基づく変更であり、質問の表現が変わったために次の担当を変更したものではない。

## 付録C：参照資料

以下のrepositoryリンクは、特記がなければレビュー固定ref `d6510db9326e9335bedd03d0d07c490561e23112` を指す。S01〜S18は本レビューで直接本文または関連範囲を取得して検討した。S19〜S24は資料公開・来歴確認および結果の関連索引、S25〜S27は研究の継続性を確認する既存資料である。リンクの記載だけを再実行や完全再検証の意味にしない。

- **S01**：[H4後科学レビュー案 v1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_post_scientific_review_v1.md)
- **S02**：[不足検証11項目](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_validation_gaps_v1.md)
- **S03**：[H6技術pilot契約案 v1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h6_pilot_contract_draft_v1.md)
- **S04**：[H4 v5結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md)
- **S05**：[H4 v5 science controller](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_h4_science_v5.py)
- **S06**：[Native DF lowering v5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py)
- **S07**：[対称controlのstage計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_control_plan.py)
- **S08**：[State-action・finite Horner・sector接続](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2a_state_action.py)
- **S09**：[現行native DF synthetic tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py)
- **S10**：[旧native DF synthetic tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/tests/tracks/resource_applicability/test_ax2a_native_df.py)
- **S11**：[入力・参照の一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json)
- **S12**：[B0 q4 correctness一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B0_q4_correctness.json)
- **S13**：[B2 K2 correctness一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B2_K2_correctness.json)
- **S14**：[B3 K6 correctness一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B3_K6_correctness.json)
- **S15**：[B1 S4 q4 correctness一次記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B1_S4_q4_correctness.json)
- **S16**：[H6機械可読契約案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json)
- **S17**：[DF Hamiltonian・builder・solver](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/df_hamiltonian.py)
- **S18**：[分子別DF rank設定](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trotterlib/config.py)
- **S19**：[GPTレビュー資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_gpt_review_index_v1.md)
- **S20**：[Source補完・公開対応manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_completion_v1/source_publication_manifest_v1.json)
- **S21**：[凍結source情報](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/source_freeze_v5.json)
- **S22**：[保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/saved_evidence_audit_v5.json)
- **S23**：[全wrapper費用集計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/wrapper_cost_summary.json)
- **S24**：[実行terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/terminal_status.json)
- **S25**：[AX-1b後の既存研究方針記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax1b_post_scientific_review_2026-10-09.md)
- **S26**：[AX-0 benchmark契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax0_benchmark_protocol.md)
- **S27**：[原論文と現行実装の既存対応表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax0_model_correspondence.md)

### 外部一次資料

- **E01**：[Günther et al., Phase estimation with partially randomized time evolution, arXiv:2503.05647v2; PRX Quantum 7, 020332 (2026)](https://arxiv.org/abs/2503.05647v2)
- **E02**：[Simon and Love, Halving the Cost of Controlled Time-Evolution, arXiv:2511.13855v1, Sec. III](https://arxiv.org/pdf/2511.13855)
- **E03**：[OpenFermion v1.6.1 low_rank.py — 実行版に対応する一次source](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py#L76-L169)
- **E04**：[SciPy 1.14.1 expm_multiply 公式API](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.sparse.linalg.expm_multiply.html)

外部資料は2026-10-10時点の取得内容を参照した。PR原論文はv2の版情報、control論文はv1の本文、OpenFermionはv1.6.1のsource、SciPyは1.14.1 APIに限定する。control論文のPDFは本文抽出を使用し、図のスクリーンショット取得が失敗したため図に依存する新しい主張は行っていない。repositoryの実装と独立した代数的確認を本書の中心根拠にした。

---
**記録上の終了**：独立科学レビュー完了。Codexによる限定検証・H6準備へ進む研究方針を提示。新たな分子科学実行・H6 launchは未認可。
