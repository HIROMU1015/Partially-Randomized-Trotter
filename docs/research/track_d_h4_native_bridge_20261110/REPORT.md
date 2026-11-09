# 保存H4 q4_sample00：Native Streaming＋P0 Bridge限定移送結果

2026-11-10 JST。**保存rawから固定した予測は、Strong／ReferenceのX/Y計4 wrapperで、保存P0後の全7資源・28値すべてと完全一致した。** 最適化ruleの修正・価格補正は0。4並列の予測処理全体は24.957秒、workerごとのpeak RSSは276.137〜280.805 MiBだった。一般P0／長q／全sampling lawへの資格は取得していない。

[28値の比較](H4_RESOURCE_COMPARISON.md)／[時間・記憶量](PERFORMANCE.md)／[次段階案](NEXT_EXECUTION_PROPOSAL.md)／[実行前PLAN](PLAN.md)／[STOP](STOP.md)。

## 固定条件と原本

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、元Hamiltonian H／number-CDF8／元frame順LD2、ordered partial S2/s1/r32/K0／h=δ=.1。保存q4_sample00、complete step2後global cut、C1 head-tail追加融合OFF。元Hの厳密基底状態を仮定し、状態準備コストは除外する。今回ground stateは求めず、state-vectorを使わない。

既存Gaussian共有・head relative・tail basis-run共有・step境界融合・scalar/identity・actual adjoint・真空χ・二CSWAP cascade・X/Y readoutを含む保存raw bytesをそのまま使用した。新しい構成・cut選択・回路生成はない。P0条件はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1.0／qubits initially zero=False／topology-free／既存宣言global phase保持。4件の保存compile recordと最終比較のprices側profileで確認した。

| identity | 確認 |
|---|---|
| C6a実行SOURCE | 6b6667026ec8b670f610e41141d195883a9eb513 |
| 原本archive | 6,950,831 bytes、20 member、SHA256 `a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b` |
| events.json | SHA256 `919f2e784920581380cb4090c59fff34741d914e9d340d249bca554eba8cadb1`、既存checked_recordで4 step×r32・SOURCE／endpoint／packet／law／固定cutを検査 |
| raw4／compiled4 | 公開payload_archiveの各member hash／sizeと一致。原本からread-onlyで取得、展開保存・再生成なし |
| 既存再利用証拠 | 2f316fb229149117177367720a6eaa5aa8a94b76の中央cut再利用結果を参照。旧PASSは保存し、新しい全入力correctnessとして数えない |

## 予測と照合を分離した来歴

既存リソース評価branch `codex/resource-eval-track-d-longq-structure-20261109`を継続。基点8b6ee8677d447ddc22d1587acdbe85b343ec7d58。

1. 実行SOURCE **7ff28615a3a7491a5a027e961d1e26fab583aebf** で、既存native lowerer／T0／1Q bridge／CX bridgeを無変更で呼んだ。新runnerはI/O、時間・frontier観測、比較を担当する。
2. raw4だけから予測し、**6049a71bddbf6cd24fdac5c78073d54588ccef51** でPREDICTIONS.jsonと4個別結果を固定した。PREDICTIONS.json SHA256は `5df5d27b00399eb14b4d0dcfe87e1ea5fadce270c81ee28926761312ffbaf77d`。
3. その後、別compare処理でcompiled4／pricesを開き、再計数と予測を比較した。参照値は利用者の指示等で事前に知られていたため盲検とは称さないが、予測関数へ価格・compiled列・補正定数は渡していない。
4. 比較の最初の結果をeb660f3515dd3645b6eb3a4a6ee0dc3acd0d620eで保全し、同commitの別diagnosticだけで保存予測word／compiledのwire順・角・phaseを調べた。再最適化・予測変更は0。

compare CLIのcommit引数は、その時点の `HEAD`＝6049a71bだった。最初のCOMPARISON／STOP内の文字列HEADを改変せず、[DIAGNOSTICS.json](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/DIAGNOSTICS.json)に解決済み40桁SHAと予測／比較hashを記録した。将来のHEADを参照する意味ではない。

## 結果と資格

| 方式 | 軸 | 価格一致 | Native命令数 | 反復数 | 最終size |
|---|---|---|---:|---:|---:|
| Strong | X | 7/7 | 200882 | 10 | 42839 |
| Strong | Y | 7/7 | 200887 | 10 | 42839 |
| Reference | X | 7/7 | 203178 | 10 | 42562 |
| Reference | Y | 7/7 | 203183 | 10 | 42562 |

T0 counterとmax-plus transferの全7指標も一致した。未対応gate遭遇0、価格残差0、価格に合わせるrule追加0、閾値緩和0。Referenceの追加8 qubit・CSWAP・χ・readout・cutの融合損失も、保存full wrapper由来のnative列に含まれている。RZをT、topology-free depthを実時間へ変換しない。

完成wrapperのwireごとのopcode／wire順は、disjoint命令の並べ替えを除いて保存P0と一致した。RZ角の最大差はStrong5.3291e−14 rad、Reference1.9540e−13 rad。宣言global phaseのmod2π差はStrong4.4946e−9 rad、Reference4.5615e−9 radで、2πのRZ表現変更による補正は0だった。

このphase差は、約−9万radまでbinary64で逐次集約する新stream側と、既存compilerのMPFR160宣言位相集約との数値差が有力な原因である。今回その寄与を厳密に分離・修正したとはしない。**完成wrapper全体のglobal phaseはX/Y確率に影響しないため、この差だけで相対χの誤りや測定不一致とは判定しない。** 一方、価格一致を新しい位相付き作用PASSへ昇格させず、新しい全入力simulation／observable照合は0。旧人工の精度条件を緩めてH4をPASSにしていない。

## 再利用・回帰・実際に省いた処理

新しい最適化器／Gaussian合成器は作っていない。標準gate definition＋公式2×2 ZSXX、strict-shorter 1Q再合成と両wire隣接CX相殺を再利用した。観測subclassは元feed後に記憶量を読むだけで、人工2件で同じword／価格／phase／反復を確認した。旧47件もPASS、重複なし合計49人工・技術回帰。旧保存54人工caseのPASS結果とSOURCEは8b6ee86からbyte不変で維持し、54を新実行件数として二重計上しない。

新H4 whole-wrapper構築・CAR dense target・P0 transpile・state simulationを省いた。ただし保存raw全体をQPYからQuantumCircuitへ読み込んでおり、QC保持が完全になくなった結果ではない。標準gateの小definition／1Q合成QC、約20万native命令のwordとbridge履歴も保持する。旧pipelineのSOURCEを復元して新whole-pathを構築する処理は呼んでいない。

## 判断材料と保全

推奨は**構造評価器を限定的に継続する案1**。固定実H4で28値が一致し、広範なcompiler再実装を追加せず到達できたためである。宣言phase精度の整理は限定課題として残す。今すぐ全P0互換を追求したり、q1024〜4096へ進んだりする判断ではない。次の対象でruleが増殖するなら、部分構造＋既存A/F/SとPR共同最適化へ重点を移す案3を比較する。

新sampling／H4生成／P0／全入力simulation／ground solve／DF refit／別path／別cut／q64以上／fit／N／全13-round更新／詳細科学レビューは0。A/F/S・既存208−224qの証拠階層は不変。旧SOURCE・input・raw／compiled・結果・grant／ledger／marker／STOPを保持し、native角列もprivate localだけに保存する。[SOURCE](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/SOURCE.json)／[PREDICTIONS](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/PREDICTIONS.json)／[比較](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/COMPARISON.json)。公開ref／tree／bytes照合後mandatory STOP。次の担当はユーザー／GPT。
