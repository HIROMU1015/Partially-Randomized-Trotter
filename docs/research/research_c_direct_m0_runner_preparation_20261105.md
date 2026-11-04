# 研究C：固定1-word・m0のDirect信号検証runner準備

2026-11-05 JST。**技術実装と人工検証を実施。保存実H4の科学数値再評価は0、契約案はNOT_READY、MANDATORY STOP。**

Branch：`codex/research-c-direct-round-m0-runner-preparation-20261105`。
独立worktree：`/tmp/prt-c-direct-round-m0-runner-preparation-20261105`。
起点は公開済みDirect評価器 `db6313156f97a37c2710f56ed7a93a34967d58f4`。
[前段報告](research_c_direct_round_rigorous_signal_readiness_20261105.md)と旧SOURCE/契約/raw/STOPを保持し、追加moduleで準備した。
正式共通指示2026-10-25の§8.1・§8.4・§10を適用。科学実行・重要科学レビューへは進めず、承認証明や原文保存は作らない。

## 固定対象と今回の到達点

将来の対象は、直鎖H4・隣接1.00 Å/STO-3G、number CDF8/元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1。
Nα=Nβ=2、sector36/system256/controlled512、own-DF ground oracle、状態準備費用除外、既存P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx・MPFR160 declared phaseを保持する。
Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
今回はm0・1 occurrenceだけ。WordはStage Pと同じF6:0／F7:6,7の交互16回、32 eventである。

保存されたcompiled/raw/actual-inverse gate record、literal block reference、compiled head/X/Y shellを再利用する。
新しいword、sampling、native build/compile、literal reference再生成、旧Stage P/7-call再実行、B2再解析、期待bias校正、RPE shots/energy、Stage C、actual CAは含めない。

今回追加したもの：

- [oracle距離primitive](../../src/trottertracks/c_direct_m0_oracle.py)：exact real-symmetric Hの固有値分離と状態距離。
- [固定m0 runner](../../src/trottertracks/c_direct_m0_runner.py)：保存入力・状態・shell・forward/inverse・X/Y・partial/STOPの接続。
- [passive入口](../../scripts/prepare_research_c_direct_m0.py)：bytes/JSONだけでSOURCE・入力・未確定項目を検査する。
- [人工テスト](../../tests/management/test_research_c_direct_m0_runner.py)、[未承認契約案](research_c_direct_m0_contract_unapproved_20261105.json)、[technical artifact](../../artifacts/research_c_direct_m0_runner/2026-11-05/technical)。

旧Direct評価器、Arb/gate/layout、Full512 operator評価器、他Codexのmoduleは変更していない。
固定1-wordをfresh IID標本へ偽装せず、専用の`FIXED_SAVED_WORD_M0` scopeを使う。

## 初期状態距離の保証と未確定のtarget

既存のown-DF oracleを維持し、保存vectorを別のexact reference stateへ採択し直していない。
保存vector vのexact binary64値をcentreとして使い、正規化したground state gへのvector2距離ηを別に包囲する。
費用除外の状態準備をη=0の根拠にはしない。

準備したprimitiveは**与えられたexact real-symmetric H**に対して次を行う。

1. NumPy固有値計算1回で、最小二固有値の間の候補σを提案する。この出力は数値候補で、gap保証ではない。
2. Exact rational LDLᵀでH−σIのinertiaを求め、負1・正d−1・零0を確認する。Congruenceによるinertia保存からλ₀<σ<λ₁を保証する。
3. 元のvからexact rationalのnorm²、Rayleigh商μ、正規化残差²r²を計算する。
4. μ<σかつr²/(σ−μ)²<1を確認し、`η ≤ | ||v||−1 | + sqrt(2) r/(σ−μ)`をArb192で外向きに包囲する。

根拠は、u=v/||v||のexcited成分について`||P_exc u||≤r/(σ−μ)`、phaseを合わせたunit ground stateとの距離について`||u−g||≤sqrt(2)||P_exc u||`である。
正規化はこの証明内の量で、伝播centreのvを書き換える操作ではない。
Global phaseを合わせる自由度は信号の二次形式を変えない。
Exact LDLのzero pivotは、matrixが非特異でも起こり得る。この実装はそこでSTOPし、pivotの摂動、正則化、別shiftへのretry、gapの捏造をしない。

**残る科学条件は、保存sector Hのexact bit値が既存own-DF oracleの数学的Hamiltonianそのものか、というtarget同一性である。**
現在の実装から保存Hのbytes/hashとproducerは追えるが、exact input-H groundと研究上のown-DF groundを自動同一視する宣言はしない。
また、保存Hの実配列は今回decodeしていないため、そのexact symmetryも未検査である。
非real／非exact-symmetricならSTOPし、(H+H†)/2、片三角からのcompletion、別のCAR targetへの置換は行わない。
NumPyのHermitian固有値計算には使用する三角部分の指定があるため、数値solverの結果だけでは入力全体のexact Hermitian性を保証しない。[NumPy 1.26公式資料](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.eigvalsh.html)

このため、primitiveのcertificateは`ground of THIS exact symmetric input matrix only`と明記し、`existing_own_DF_target_identity_established=false`を維持する。
契約の`oracle_target_identity`はnull。必要な意味が元の研究定義と同じと確認できた時だけ、将来の固定科学契約へ条件を記載する。
異なるHamiltonianのbridgeが必要なら15-call範囲を黙って広げず、その追加準備と実行範囲を別に具体化する。

## 入力とSOURCE

[入力一覧](../../artifacts/research_c_direct_m0_runner/2026-11-05/technical/input_inventory.json)で、13 fileのpath・bytes・SHA256と役割を固定した。
旧Stage P契約に記載された同一ファイルのhashとも照合した。科学配列NPZ、QPY、Arb blockの評価は実施していない。

| 入力group | file数 | 役割 |
|---|---:|---|
| 保存gate/reference metadata | 6 | raw、compiled、actual inverse、same-forward inverse binding、literal reference、per-word binding |
| 保存NPZ | 4 | candidate J/U、sector J/U/H、physical sector indices、saved ground vector |
| compiled QPY shell | 3 | head/X/Y。将来の許可後にだけdecode |

Literal target IDは`a62d9d2f7036c0d7e8aa25323acf05d98c20f368cf4863d08a15cd24c62c5021`。
J/Uはcandidateとsectorのexact配列一致、およびliteral referenceにbindされたarray bytesに照合する。
Word順序・angle・gate/phase・register・actual inverseのparentを同じ保存metadataに照合する。
初期状態は`bit_reverse(saved_interleaved_sector_indices)`でsector36からsystem256へ埋め、全controlled512出力を伝播する。Gaussian途中のsector射影はしない。

[technical SOURCE](../../artifacts/research_c_direct_m0_runner/2026-11-05/technical/SOURCE.json)は新実装と関連再利用処理の識別子で、科学実行許可の証明ではない。
科学SOURCEは未固定のためnull。旧科学SOURCEの付け替え、新grant、承認記録、旧残callの転用はない。

## 準備した15 grouped calls

| call | 数 | 内容 |
|---|---:|---|
| metadata_load | 1 | 同じ保存gate/literal/inverseのJSON |
| arrays_load | 1 | 保存candidate/H/ground/basisのgroup読込 |
| target_identity | 1 | exact candidate・word・target・register・Hの構造検査 |
| shell_decode | 1 | 保存compiled head/X/Yをgroup decode |
| oracle_proposal | 1 | 固有値分離候補を生成する数値固有値計算。新ground vectorは解かない |
| oracle_certificate | 1 | exact inertia・残差・正規化距離・ηの包囲 |
| sector_embedding | 1 | 既存basis順序の埋込み |
| state_binding | 1 | 元centre＋ηをSOURCE/targetへbind |
| literal_prepare | 1 | 保存block包囲のunpackとnorm-growth準備 |
| head_action | 1 | compiled／ideal headを各一回 |
| forward_action | 1 | 同じ一wordのnative/literal作用 |
| inverse_witness | 1 | 同じforwardのactual inverseと数学的adjointの状態作用差 |
| readout_X / readout_Y | 2 | 両axisの信号区間と差、norm/radius |
| summary | 1 | 固定m0結果・未資格項目・STOP |
| **合計** | **15** | **将来の科学実行候補。今回の実行数ではない** |

Signalはcompiled shell込みz_Cとideal-shell literal z_Tのsigned X/Y差。
Arb192、complex rectangleのexact rational endpoint、初期状態η、literal非unitary norm成長を保持する。
Shellは直接作用させるのでStage P shell誤差を重複加算しない。分子Head/B2/categorical transportはこの範囲に足さない。
Inverseは同じforwardのstate witnessであり、Full512 operator/inverse保証ではない。
旧公開`c_arb_column_batch.reduce_pair`によるexact common-unitary除去を再利用し、共通prefixを一度だけstateへ作用させ、normに影響しない共通suffixを除く。位相は保持する。

出力は固定1-wordのみで、fresh IID sample、期待bias、統計confidence、校正資格、RPE accuracy、actual CAを返さない。
N、δ_cal、δ_compile、failure/abort許容値はこの固定word検証では採択せずnull。

## 性能、保存と人工検証

保存gate JSONを読むだけの[静的workload](../../artifacts/research_c_direct_m0_runner/2026-11-05/technical/static_workload.json)では、compiled/actual inverseは各11386 gates。
全512列でなく、一つの明示状態について局所Arb block作用を用いる。Native forwardはO(G·512)で、gateの順序依存がある。
Inverse診断には上記の公開済み因子削減を適用する。Literal referenceは再生成しない。
実H4の秒数・実RAM・speedupは新測定していない。

今回の人工テストは小さい逐次依存問題であり、process並列の追加より既存FLINT局所作用とexact rational処理が適する。
実行時は利用可能なCPU affinity/quota、空きRAM、他jobへの影響を再確認する。
CPU/RAM/worker/wall capを新設せず、共有環境・他jobのprocess/cache/lockを変更しない。

人工・回帰テストの正本は[tests.txt](../../artifacts/research_c_direct_m0_runner/2026-11-05/technical/tests.txt)。
既知の非対角rational ground、complex global phase、非正規化centre、exact inertia、zero pivot/縮退、wrong target/word/phase/register/basis、shell decode、15-call人工end-to-end、partial/例外/割込み/STOPを含む。
新規26件＋前段Direct評価器46件、合計72 PASS。最終ログは`technical/final_tests.txt`（初回71 PASSログも保持）。前段Direct評価器の人工回帰も実施した。ローカル人工検証で、実H4の新結果やCI・外部独立再現ではない。

将来の科学出力には、oracle候補・certificate、embedded状態、head/forwardのcentreとradius、X/Y各結果、exception、calls/STOPを保存する。
途中失敗時も完了済みファイルを保持し、fresh outputを再利用しない。Retry/resume=0。

## 非実行状態と次担当

Passive結果は`NOT_READY`。未確定は、非実行draft、固定scientific SOURCE、保存Hとown-DF oracleのtarget同一性。
CLIはpassive専用で、科学実行flagを公開していない。Libraryの`run`も明示的な実行要求がなく、またはdraftが未確定なら、配列/QPY decode前に拒否する。
これらは既存科学条件を確定するための境界であり、ユーザーの承認を証明・保存する手続きではない。

次担当は**ユーザー／研究C**：own-DF oracleの数学targetと保存Hの関係を確定する。
**Codex**：同じtargetだと確認できた場合、その条件・scientific SOURCE・15-call範囲を固定した科学実行案を提示する。異なる場合は必要なbridgeを具体化する。
科学実行・重要科学レビューはそれぞれ別の明示許可を受けるまで行わない。

技術成果をcommit/pushし、remote SHA/tree・報告本文・主要blobを照合する。
**実H4科学実行0・重要科学レビュー0。公開照合後MANDATORY STOP。**
