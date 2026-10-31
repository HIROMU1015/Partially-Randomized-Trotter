# 研究C：保存3点のRC2／PR一次診断・14-call実行結果

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**固定SOURCEを変更せず14/14 logical units COMPLETE、correctness FAIL0、新fit／optimizer iteration／retry／resume0で完了した。** 初期penalty寄与一致は勾配一致ではなく、両保存終点は登録engineering gtolを満たさない。RC2終点には、全tensorを保持する既知の方向でL2項を下げる一次成分が残っていた。「構造上Pをこれ以上下げられなかった」とは説明できない一方、最適ρや97 iterationで停止した歴史的原因は未特定である。

## SOURCE・契約・対象

| 固定物 | Identity／条件 |
|---|---|
| 科学実行SOURCE | [0996a5abb4876c2e450439044c26e7df61dfd04c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0996a5abb4876c2e450439044c26e7df61dfd04c)、無変更で一度実行 |
| 数値module/entry SOURCE | a41329d7e860415ddfc21df5b02728804846d16d、最終test/manifestは上記公開commit |
| [固定契約](research_c_regularization_first_order_contract_unapproved_20261101.json) SHA256 | b7820771dbb67de704348867745b268e60b5787a27c109961ae5e6a1102f727b |
| [SOURCE39](../../artifacts/research_c_regularization_first_order/2026-11-01/technical/SOURCE.json) SHA256 | 328a84c93b89e1cf02fd23784fa53f0435b857ce490c6996713b5fd7416c47fd |
| 分子・表現 | 保存直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2・sector36、通常number CDF rank8・一般対称J・実直交U |
| 保存規約 | 元frame順、LD2のtail、元g補正κ・scalarを変更しない。今回Hamiltonian／sector／PF evolutionは構築せず、δ・時間窓は非適用 |
| 目的 | base=f/f*、RC2=base+.1P/P0、PR=base+.1(Mε/M*)²。Pは全frame ordered空間J²、PRは元順tailの非identity Z/ZZ |
| Scale | f*=2.1839632053132894e-9 Ha²、P0=4.410884790420646 Ha²、M*=2.2093719370722074 Ha、ε=1e-8 Ha、既存ρ=4.951304123962429e-11を変更しない |
| 評価環境 | NumPy1.26.4／SciPy1.14.1／ffsim0.0.83／JAX・jaxlib0.4.38、JAX x64、既存fit venvを読み取り専用で使用 |
| Fresh output | [run01原本](../../artifacts/research_c_regularization_first_order/2026-11-01/run01/) |

3点は共通qualified初期x0、RC2最終x、PR γ=.1最終x。元RC2 fit SOURCEは150f904f6975868abe09966dda8ab41722e2a7cc、元PR fit SOURCEはe1085c95f5c01b68566e17d543ace5832cb94061であり、新評価SOURCEと独立に保持した。9入力のhashは契約に固定。初期／RC2／PR exportはそれぞれ78f67ec5…、49721748…、f47a9fbf…で、完全hashは契約参照。元P-H4来歴8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50を保持する。

[技術診断・数学・比較政策](research_c_regularization_strength_stopping_diagnostic_20261101.md)に従い、既存ffsim loss/ADと両objectiveをそのまま呼んだ。新しいloss／codec／solverを作らず、旧fitを再開しなかった。今回のチャット許可に基づき実行し、歴史的unapproved filename/metadataは変更せず、承認証明・承認原文保存・新grantは作成していない。

## Gate・call・同値性

| 処理 | Units | 結果 |
|---|---:|---|
| SOURCE／契約／入力／環境のpassive preflight | 0 | PASS、SOURCE39・入力9・ffsim固定2ファイル一致、未使用run01 |
| 実行時人工試験 | 0 | 13 PASS、実H4を読まない。準備時123件を今回すべて再実行したとは扱わない |
| 保存snapshot読込 | 1 | COMPLETE、gと3点、再初期化なし |
| parameter fidelity | 3 | すべてPASS。保存xの既存unpackに対しJ exact、U maxabs差0、直交性gate PASS |
| base／RC2／PR value+gradient | 9 | 全9評価COMPLETE、finite float64 gradient、各vectorを保存 |
| 成分診断 | 1 | COMPLETE、L2上三角diag2／off4の解析勾配一致、両penaltyの直接U成分0 |
| 合計 | **14** | **14 attempts＝14 COMPLETE、FAIL／inflight0** |

[Preflight](../../artifacts/research_c_regularization_first_order/2026-11-01/execution_preflight.json)／[人工log](../../artifacts/research_c_regularization_first_order/2026-11-01/execution_synthetic_tests.txt)／[call journal](../../artifacts/research_c_regularization_first_order/2026-11-01/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_regularization_first_order/2026-11-01/run01/STOP.json)。保存vectorは各128成分、回転48＋J上三角80の同じ座標である。

人工13件には人工二次関数での実SciPy反復も含む。ここで報告する新fit／optimizer iteration0は実H4の科学実行についてであり、人工optimizer試験まで0と称するものではない。

DF予算CA/8=.00019875 Haの旧3点PASSは保存結果を再利用した。元のengineering guardを厳密保証へ昇格せず、新たな独立tensor norm・sector norm・physical Hamiltonian・groundを計算していない。既存loss内のtensor再構成は9評価に含まれる。

## 保存値と勾配成分

以下は今回の[結果JSON](../../artifacts/research_c_regularization_first_order/2026-11-01/run01/result.json)と9件のvalue JSONからの転記。表示の桁を丸めた表であり、原本には完全値と9本のgradientを保持した。表のnorm・比・内積・方向微分はすべて登録component_diagnostic内で得たもので、報告作成時に追加gradient／Hessian／SVD等を計算していない。

| 保存点 | base value | RC2 value | PR value |
|---|---:|---:|---:|
| 共通初期 | 1 | 1.1 | 1.0999999999999999 |
| RC2終点 | 5.592154920733334e-9 | .10000199116572644 | .1000353069637964 |
| PR終点 | 9.04433906231226e-9 | .1013632898409631 | .04262376967322519 |

異なる目的のvalueをそのまま性能順位にしない。今回再評価したRC2／PR各終点の自己目的gradient normは旧保存diagnosticの値と一致した。Cross-evaluationは同じgeometryで両penaltyを見るためであり、新候補の作成ではない。

| 保存点 | loss gradient L2 | weighted RC2 penalty L2 | weighted PR penalty L2 | RC2 penalty/loss | PR penalty/loss |
|---|---:|---:|---:|---:|---:|
| 共通初期 | 32834.019268206124 | .12388441440725963 | 1.2559620851911413 | 3.773050548435888e-6 | 3.825185320541357e-5 |
| RC2終点 | 1.57353794401662 | .12388554756581056 | 1.2561837515530925 | .07873057528538509 | .798318055392139 |
| PR終点 | 4.72393571512427 | .12445976315203183 | .6487550988526027 | .026346625072301125 | .13733360019602558 |

**初期のpenalty寄与=.1一致は勾配影響一致ではなかった。** 現座標ではPR側の初期penalty勾配が大きく、どちらも初期loss勾配より小さい。ただしこの座標依存の局所値だけで、適切な新ρや「強度不足が唯一の原因」を選べない。RC2終点ではlossとL2 penaltyのgradient cosineは−.17696931705538343、PR終点ではlossとPR penaltyが−.14661398890166502で、一部の競合はあるが停留点を示す相殺には至っていない。

自己目的の終了gradient∞は次の通り。

| 終点／自己目的 | gradient∞ | 固定gtol=1e-5＋engineering margin |
|---|---:|---|
| RC2／RC2 | .4222446600957083 | OUTSIDE_ENGINEERING_GTOL |
| PR／PR | 1.4825504139166679 | OUTSIDE_ENGINEERING_GTOL |

全3点の両objectiveも同判定だった。Gradient marginは事前固定atol1e-8・rtol1e-7で、事後変更なし。Optimizer success／旧relative-reduction停止とこの判定は別であり、旧successをFAILへ書き換えない。非zero直接U勾配をpenaltyが打ち消すこともなく、loss側U-gradient L2はRC2終点.49409447544480783、PR終点.6354665358872548だった。

## Tensor-null方向で分かったこと

事前設計通り、s_t=ΣijJ_tij、a_t=−(s_t−平均s)、dJ_t=a_t11ᵀ、dU=0の方向だけを検査した。Σ_ta_t=0なので、数学的には全tensorを保つ方向である。これを用いた新しいJ/U、有限step、fit、PF fragment変更は生成していない。

| 保存点 | 方向norm | loss方向微分 | RC2 penalty方向微分 | PR penalty方向微分 |
|---|---:|---:|---:|---:|
| 共通初期 | 21.232134063874625 | −1.4551915228366852e-11 | −2.0440502907106106 | 3.2861030529817006 |
| RC2終点 | 21.232167808987725 | −1.3322676295501878e-15 | −2.044056788098623 | 3.2865096018125692 |
| PR終点 | 21.681697552308137 | 1.7763568394002505e-15 | −2.131527034080316 | −.8057789993625196 |

方向は正規化していないため、異なる点の数値の大きさを改善量や速度として比較しない。lossの小残差も丸めを囲った厳密enclosureではない。Tensor-nullという代数的性質と、その既知方向に沿った今回の数値一次診断を区別する。

**RC2終点では固定tensorのままL2を下げる方向が残る。** 従って、少なくともこの終点について「tensor精度を保つとPはこれ以上動かせない」という説明は支持されない。同じ方向でPR penaltyは増えるため、L2減少をtail目的改善と同一視できない。PR終点にも自身のpenalty下降成分が残り、PR側だけ十分に最適化済みと仮定する根拠にもならない。

この一方向から全null空間、物理sector誤差、finite-stepでの最良λ、native価格や将来solverの軌跡は判断しない。特に「L2を続ければ必ず資源が改善する」「PRが最適RC-DFに勝つ」とは主張しない。

## 原因候補・比較政策・次の判断

今回支持できる説明は、(1)初期の勾配影響は一致していない、(2)両保存終点は共通一次基準を満たさない、(3)RC2の既知tensor-null方向にL2下降余地があり、同方向のtail目的は競合する、の三点である。強度・停止・構造の区別が具体化したが、旧97 iterationのline-search／trajectoryは復元していないため**歴史的主因の一意判定はUNRESOLVED**。最適強度もnullのままである。

現RC2／PR結果は初期寄与一致の固定pilotとして有効で、強い最適化済みbaselineの比較とは呼ばない。共通DF予算をadmissionにするB4方針と、両armの共通一次精度を確認する必要性が残る。勾配一致のために新ρを自動計算・採用したり、L2側だけ追加fitして旧PRを十分収束済みとして扱ったりしない。

**次の判断はユーザー／研究Cが担当する。** 強い対照の資格を優先するなら、同じ固定strengthの両armで停止精度をどう確認するかを先に決める。Codexの次作業案は、その方針が依頼された場合に共通停止政策の最小技術準備を行うこと。新fitの回数・停止規則・新SOURCEと契約は未採択／未作成で、この14-callから実行権限は移らない。現在の候補をStage Bへ進めるなら固定pilotとしての下流測定に限定し、そのための別準備と科学許可が必要。

## 資源・保全・公開

実行前affinity32 CPU・MemAvailable56,281,481,216 bytesを確認。既存JAX CPU value/gradientの構造別JIT/cacheを9小評価で再利用した。最初のFAILで停止できる固定順序を維持し、process増殖や重複compileを避けた。Tiny BLASのみ自process内1 threadでJAX poolと協調し、人工的なCPU／RAM／worker／wall上限は追加していない。

実行logにはCUDA-enabled jaxlibがないというJAX通知があるが、準備済みと同じCPU環境で一度実行した。環境install・SOURCE変更・別backend再実行はしていない。[実行log](../../artifacts/research_c_regularization_first_order/2026-11-01/execution_run01.txt)。

[資源記録](../../artifacts/research_c_regularization_first_order/2026-11-01/run01/resources.json)はrunner wall .9781542560085654秒、process CPU 2.17350761秒、peak RSS387852 KiB。Wallはrun内部の計測区間で、Python起動全体のwallとは区別する。

[Postflight](../../artifacts/research_c_regularization_first_order/2026-11-01/execution_postflight.json)で、SOURCE39/input9・固定契約／runコピー・14 attempts/COMPLETE・9 gradientのheader・raw27件・旧SOURCE／結果／STOPを含む126ファイルのGit blob不変を照合した。旧17-call run01、旧fit exports、H2/H4/H6のraw、grant／ledger／marker／STOP、共有／既存venv／他jobは変更していない。旧科学文書は保持し、今回の新結果だけを研究記録へ追記する。公開時にremote SHA/tree・本文・新rawを照合する。

次は実行履歴であり、消費済みrun01の再起動指示ではない。

~~~bash
PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_regularization_first_order.py \
 --contract docs/research/research_c_regularization_first_order_contract_unapproved_20261101.json \
 --execute --output artifacts/research_c_regularization_first_order/2026-11-01/run01
~~~

**14-callを完了し、公開照合後mandatory STOP。** 新fit、強度変更、再最適化、rank探索、独立tensor/sector/ground、PF/RTE、実回路build/compile、shots、CA／総費用、重要GPT科学レビューへ自動進行しない。
