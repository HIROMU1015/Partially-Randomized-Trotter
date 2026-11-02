# PR-2 V4/S1′・development資源比較 実行許可 amendment v5

日付: 2026-09-28
系列: `pr2-rebaseline-de7a5492-v1`
状態: `PR2_V4_THEN_S2_DEVELOPMENT_AUTHORIZED_MANDATORY_STOP`
held-out 1.30 Å: 未開封を維持、S3未承認

## 1. 判断と位置付け

本書は、V1--V3結果を確認した後の結果前authorizationである。既知の結果は、固定development
snapshotのintegrity/model/partial構造が通過し、rank 3/6/9のB2-GとB2-Wがordered prefixまで一致した
ことだけである。signal、controlled wrapper、compile、32/128 trajectory expected cost、resource winnerは
未評価である。

ユーザーが提示した研究判断をreview inputとして採用する。ただし、これは独立外部再現やblind reviewとは
呼ばない。判断は、PR-2を「新しいgeneration-prefix法」ではなく、**DF-prefix部分ランダム化が
finite-RTE、controlled wrapper、normalization、shot、compiled costを戻したときに中間splitとして
資源frontierへ残る条件を調べる研究**として扱い、次の一点まで進めることである。

```text
V4/S1′ correctness
  -> V4 pass時だけdevelopment 1.00 Åのshot込みresource比較
  -> mandatory STOP
  -> 研究方針を再判断
```

V4通過だけでは停止せず、本書によりS2 developmentを条件付きで事前承認する。S2終了後は結果にかかわらず
停止する。held-out H4 1.30 Å、S3、追加geometry、H12、長RPE、precision sweepは未承認である。

## 2. 固定入力と既知証拠

- development: H4 linear 1.00 Å、STO-3G、8 qubit、4 electron、DF rank 12
- development snapshot SHA-256:
  `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- Hamiltonian hash:
  `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- held-out H4 1.30 Å file SHA-256:
  `ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`
- V1--V3 result fingerprint:
  `b210b394e9cd5a8eded947b0fd12cefe19ce9f27eb6b8140b3e863df73ea7961`
- V1--V3 result artifact SHA-256:
  `0eb22c813eb838169eb455334146140467ebbc5636db78bd923b1e6bdaed46d8`
- B2-G/B2-W: rank 3/6/9でordered indicesが一致するため、以後は一つの`B2`へ統合する
- 旧S0 `STOP_INPUT_REPRODUCTION_MISMATCH`は上書きせず、旧S1 authorizationもfalseのまま保持する

held-outについて許す操作は、存在確認とraw file SHA-256の照合だけである。NPZ load、signal、cost、ranking、
candidate selectionを禁止する。

## 3. 共通estimand・baseline・精度

targetは固定ground stateに対する

$$
z_{12}(T)=\langle\psi_{12}|e^{-iH_{12}T}|\psi_{12}\rangle
$$

とする。比較baselineは次で固定する。

| ID | 固定内容 |
|---|---|
| B0 | generation-prefix rank 6でresidualを捨てるdeterministic $S_2$ |
| B1 | rank 12 deterministic $S_2$ |
| B2 | rank 6 DF-prefix deterministic backboneとexact residual finite-RTE |
| B3 | $L_D=0$ two-body-random endpoint。one-bodyは決定論側 |

rank 3/9はB2のcontrolであり、primary winner判定へ混ぜない。finite-RTE gridは
$r\in\{1,2,4,8,16,32\}$、$K\in\{2,4\}$、coefficient threshold 0、
`extract_identity_phase`で固定する。

complex-signal accuracyは $\epsilon_{\mathbb C}=0.05$、
$\epsilon_{\rm axis}=0.05/\sqrt2$、$\alpha_{\rm axis}=0.025$とする。primary signalは既知の
finite-distribution normalizationで補正したmeanで、raw attenuated meanも別fieldへ保存する。axis bias
$b_{j,a}$が$\epsilon_{\rm axis}$以上ならineligible、そうでなければ

$$
N_{j,a}=\left\lceil
\frac{2\mathcal B_j^2}{(\epsilon_{\rm axis}-b_{j,a})^2}
\log\frac{2}{\alpha_{\rm axis}}
\right\rceil
$$

とする。$q$ stepでは$\mathcal B_j$をone-step multiplierの$q$乗とする。shotは解析的に数えるだけで、
量子backend shotは実行しない。

## 4. V4 / S1′ correctness

developmentだけを使い、$T=\delta=0.1$、$q=1$とする。

1. B2 rank 6とB3の全12 $(r,K)$についてraw/corrected mean、normalization、attenuation、
   corrected/raw identity、finite truncation bias、outer-PF bias、Re/Im biasとshot式を検査する。
2. B2 rank 3/9は$(r,K)=(1,2)$のsentinelを検査する。
3. B0 rank 3/6/9、B1 rank 12のdeterministic signalを検査する。
4. cosine wrapperがRe、sine wrapperがImを返すこと、sine符号、`diag(I,U)` control、identity/global phase、
   measurement inclusion、state-preparation exclusionを、実回路とdense expectationで検査する。
5. random各cellはfrozen canonical 1 trajectory/axis、deterministic各baselineはexactで、
   `rz,sx,x,cx`、Qiskit 1.3.0、optimization level 1、seed 17、coupling mapなしのfull wrapperをcompileする。
6. expected-cost 32/128 MC、resource selection、10%判定、state-preparation break-evenはV4で行わない。

V4 terminal statusは次のいずれかとする。

- `V4_CORRECTNESS_PASS_S2_DEVELOPMENT_AUTHORIZED`
- `BLOCKED_IMPLEMENTATION_INVALID`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`

最初のstatusかつdeviationなしの場合だけS2へ進む。

## 5. S2 development比較

development 1.00 Åだけを使い、共通物理時間$T=0.8$、$\delta=0.1$、$q=8$とする。

- B0 rank 6、B1 rank 12、B2 rank 6の全12候補、B3の全12候補を評価する。
- B2/B3のcorrected q-step meanはfresh IID tailを各occurrenceで引く平均に対応し、one-step corrected
  operatorの8乗とする。raw mean、normalization multiplierの8乗、attenuation、bias分解を保存する。
- rank 6で選ばれたB2設定をrank 3/9へ固定移送し、control signal/costを記録するがprimary選択には使わない。
- full measured Hadamard wrapperは状態準備を除外し、ancilla準備、ordinary controlled evolution、axis rotation、
  measurementを含む。primary metricはcompiled RZ、secondaryはCX、RZ/CX/total depth、sizeとする。
- deterministic costはexact、random costは各cell 32 trajectoryのMonte Carloで直接compileする。
  S2 master seedは`20260927102`とし、cell seedはstage/method/rank/r/K/streamをcanonical JSON SHA-256へ
  入れて決める。cosine/sineは同じtrajectory列を共有する。
- RZ relative SEが2%超、または10% decision/method内settingがengineering intervalで未解決の関連cellだけ、
  別seed streamの96 trajectoryを追加して初回32とpoolし、合計128とする。追加は一度だけである。
- engineering intervalはmean $\pm2SE$（下端0）であり、formal confidence intervalとは呼ばない。
- total no-preparation workは$G_j=\sum_aN_{j,a}\bar C_{j,a}$、intervalはaxis別端点を同じshot数で合計する。
- 同一method内はeligibleなpoint estimate最小を選び、tieは小さい$r$、次に小さい$K$とする。
- primary比較ではB2のselected candidateとB0/B1/B3 selected endpointを比べ、ratio interval上端0.9未満を
  10%以上のmaterial advantageとする。
- 共通state-preparation cost $P$ RZ相当/shotを加えたpairwise break-even $P^*$をsecondary sensitivityとして
  報告する。hardware依存の$P$は選ばず、primary rankingを変更しない。

## 6. S2判定とmandatory STOP

S2はamendment v2のdecision ruleを維持し、次のいずれかを返す。

- `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`
- `S2_CONDITIONAL_RESOURCE_MAP_AWAITING_REVIEW`
- `S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`
- `BLOCKED_IMPLEMENTATION_INVALID`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`

ただし本実行の目的はdevelopmentで一度比較して研究方針を再判断することである。transfer candidateであっても
`automatic_next_stage=null`、`S3_authorized=false`とし、held-outを開かない。128後もsetting/decisionが
不確実なら追加sample、threshold変更、grid追加、precision sweepで救済しない。

## 7. 実装・証拠規約

1. 本書とmachine-readable authorizationを先にcommitする。
2. V4/S2 module、non-overwrite runner、validator、dedicated testsを別commitで固定する。
3. executable sourceがHEADと異なる場合はrunnerを拒否する。
4. V4 artifactを生成・検証し、PASSかつdeviationなしの場合だけS2 runnerを許す。
5. S2 artifact、JUnit、source hash、worktree status、全seed/provenance、32/96/128 workloadを保存する。
6. 結果をoverview、validation status、manifest、索引、研究ノートへ反映して停止する。

dirty worktreeの無関係変更を科学的証拠へ混ぜない。artifactはlocal validationであり、量子backend実行、noise、
immutable CI、外部再現、最終total-cost評価とは呼ばない。
