# H₄ M=13 Q3のサーバー2 worker検収

2026-11-22 JST、`GPUA100-gleap02`（`server2`）で、Full PF・B0I・PR-DFの保存実入力を通常workerの事前照合に通した。3件すべて `REAL_HOST_DRY_CHECK_PASS`。旧M=11・12も、中央の保存割当6件すべてが通過した。隔離キューでは人工ジョブ12件を配信・実行・回収し、同一runのResumeで追加起動0件、二重採用0件を確認した。

**RomuTailからサーバー2への14ラウンド配送は未検収。** 受信済みの割当1,323件を読み取り、Xが14要素または `horizon_risk_source` 付きの要求は0件だった。M=13の3入力は、公開済みの固定要求と元の14ラウンド結果のidentityから検収用の割当をローカルで組んだ。PR-PauliのM=13には中央の固定要求がなく、受付は未確認。新規の実H₄信号・回路build・compile・Q3は各0件。

## 固定版と実使用環境

| 用途 | 固定版 | サーバー2の実使用パス |
| --- | --- | --- |
| 科学入力のGit解決 | `28d976b99cdf8329adcc1b79327e329e947ef774` | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99` |
| M=13 worker adapter・readiness checker・隔離キュー | `9171f114c790f2642e1386b43f9549ea00f3e46a` | `$EVIDENCE/fleet` |
| 14ラウンドの評価器 | `89fcde14f6ac5f4b97be234f9b0822091e7d9ac7` | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-m13-risk-89fcde14f` |
| Full PF・B0I route | `cbc471c913072de45c25a0ed22d51dff8a9b7034` | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-route-view-b-cbc471c91` |
| PR-DF route | `23160cc876807c0300be3de811f767b5810640a7` | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-route-view-d2-23160cc87` |

事前照合・Full PF・B0I・人工キューは `/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python`、PR-DFのadapterは `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python` を使用した。両方とも Python 3.11.0rc1。PR-DFは通常adapterの `prdf_d2_split` を通り、固定AST hashを検査して `PASS_UNCHANGED_GUARD` を返した。adapterファイル・評価器・実入力の個別SHA-256は [EVIDENCE.json](EVIDENCE.json) に記録した。

## 保存実入力と人工ジョブ

| 方式 | M=13の保存実入力 | M=11・12の保存中央割当 | 通常用人工キュー | 有限支持の大ショット人工キュー |
| --- | --- | --- | --- | --- |
| Full PF | PASS、X/Y各14要素・信号14行・位相半径14個 | 2/2 PASS | 2/2採用 | 2/2採用 |
| B0I | PASS、X/Y各14要素・信号14行・位相半径14個 | 2/2 PASS | 2/2採用 | 2/2採用 |
| PR-DF | PASS、X/Y各14要素・信号14行・位相半径14個 | 2/2 PASS | 2/2採用 | 2/2採用 |
| PR-Pauli | 中央の固定要求待ち | 対象外 | 対象外 | 対象外 |

M=13の三つの入力は、公開要求に含まれた元の期待identityの代わりに、同じ固定結果に保存されたM=13の `configuration_id`・`binding_id`・`total_configuration_id`・信号locatorを検収用の期待identityにした。元の要求ファイルのバイト列と結果SHA-256を照合し、双方を保全した。X/Y配分・信号の行・位相半径は固定入力を使った。三つの固定結果の `conditional_RZ_cost` は `null` で、今回の検収用入力に終端X/Y価格は添付していない。 [保存実入力の固定検証](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ac869a3dc239cf65489aacaf200cb4f113880c28/docs/research/rpe_m13_real_validation_20261122/HANDOFF.md)

サーバー2内の隔離controllerから通常の分散executor、worker subprocess、receipt照合、採用まで通した。通常用人工ルートは人工MSEの独立計算と厳密一致。大ショット人工ルートは固定有限支持kernelで、各方式で最大1,008 shotの配分を配送した。採用後の通常Resumeは `REUSED_COMPLETED`。さらに同一の通常CLI・入出力で再入場し、job ID、attempt ID、結果SHA-256、採用記録、終了理由の一致を確認した。検収runnerの初版では基準点 `s=1` を件数の期待値に含めていなかった。期待値修正後、隔離runを新規に作り、最終証拠 `queue02` を採取した。

別に、既存の人工数値検証を固定Pythonで再実行した。m12→m13の枝選択140例（厳密tie 30、X/Yともゼロ4）が参照と一致した。通常Q3の代表例は参照MSEと上下界が厳密に一致し、表示値は \(4.5958926906144\times10^{-11}\,\mathrm{Ha}^2\)。大ショットの人工代表例では参照 \(1.1465719629344267\times10^{-8}\,\mathrm{Ha}^2\) が初回区間 \([1.1465719629342856,1.146835214266772]\times10^{-8}\,\mathrm{Ha}^2\) に入り、閾値 `Fraction("0.00159360010199040")**2` に対してPASSだった。省いた質量と丸め残差を含む検証、精密化とM=11・12の数値回帰も通過した。 [既存の人工数値検証](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d200568c5ca26b93bd97e848657a5406261c27a/docs/research/rpe_q3_m13_validation_20261122/HANDOFF.md)

## RomuTail担当への引渡し

照合した中央の公開runは `search-29795d0e2e2593819e27`、公開時のreleaseは `romutail-h4-rerun4-1355ad65fa`。サーバー2にある同releaseのprofileの `route_profiles` に `horizon_risk_commit` の登録がなく、adapterは `f67a86291a8a0247f99bd952d7ea0b41ab1a14b9` を選ぶ。同じ3件をこのadapterの `dry-check` に直接渡すと、3件とも `OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN`、科学call各0件で停止した。調査時の運用ガイドHEAD `c81bbe7d5330b25ca87b5060d0c520ea3e4654eb` のadapterにも旧horizon制限がある。中央のローカルroute指定は中央の固定記録から、外部ホストの受付は対象ASSIGNMENTとprofileから照合する。 [中央の固定運用記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/bb536f384127107006f3ab03a5c18f7a99c3de3b) · [中央M=13の接続仕様](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b9f7bf6a48a86b6638f4958d151c8d36ac99505/docs/research/rpe_m13_central_20261122/HANDOFF.md)

1. 中央の検証用runで確定した `REQUEST`／`ASSIGNMENT`、releaseと選択済みhost profileを渡す。Full PF・B0I・PR-DFをサーバー2へ割り当て、受理した要求SHA-256とjob／attempt IDを対応付ける。
2. 14ラウンド要求のrouteに固定 `horizon_risk_source` と `horizon_parent` を指定する。host profileの `route_views` に `89fcde14…` の実パスを登録し、必要な `route_profiles` を `horizon_risk_commit` で選ぶ。通常adapterは検収済み `9171f114c…` の該当実装を使う。PR-DFは固定要求が選ぶroute kindまで確認する。中央側の固定候補の構成IDがこの報告の3件と違う場合は、中央の実要求で先に通常readinessを通す。
3. 中央の検証用キューで人工ジョブを配送し、RomuTail受領・採用まで確認する。同じ通常Resumeで追加起動・二重採用が各0件であることを中央ログとサーバー2のreceiptで照合する。PBSとRomulaptopのWSLには各ホストの配置確認が別に必要。

稼働中run、運用profile、共有currentは検収中に編集していない。

## 再現と原証拠

機械可読の [EVIDENCE.json](EVIDENCE.json) に方式別の実行Python、要求SHA-256、`READINESS.json` の実パスとSHA-256、人工job・attempt ID、結果SHA-256を記載した。以下はサーバー2の専用領域で実行する。数値検証は固定参照の人工入力で、保存実入力は `dry-check` だけを呼ぶ。

```bash
export M13_SERVER2_EVIDENCE_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/_workspace/01-current/H4_M13_SERVER2_ACCEPTANCE
WT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees
PY=/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python
EVIDENCE="$M13_SERVER2_EVIDENCE_ROOT"
REPLAY=$(mktemp -d "$EVIDENCE/replay.XXXXXX")
"$PY" -B check_saved.py "$REPLAY/readiness"
"$PY" -B check_queue.py standard "$REPLAY/queue/standard"
"$PY" -B check_queue.py large "$REPLAY/queue/large"
"$PY" -B "$WT/rpe-q3-m13-validation-20261122/tests/check_rpe_q3_m13_validation.py" \
  --fixed-view "$WT/b2-route-view-b-cbc471c91" --report "$REPLAY/NUMERICAL_REFERENCE.json"
```

`check_saved.py` はローカルにM=13の中央割当が届いていれば、その利用を優先するよう停止する。実割当を受け取った場合は、固定profileで次の通常入口を使う。

```bash
"$PY" -B "$EVIDENCE/fleet/scripts/check_rpe_q3_host_readiness.py" \
  --assignment "$ASSIGNMENT" --profile "$HOST_PROFILE" --output "$REPLAY/central-readiness"
```
