# H4 保存順XDF8：表現・ground/alias・決定論S2の限定科学実行

2026-10-26 JST。**93/93 logical units COMPLETE、correctness FAIL0。表現・own-ground/aliasと26/26 PFセルがPASSし、s=1／4ともPOTENTIALLY_FEASIBLE。** 最終CA・native・合成誤差・shots・総資源は今回評価していない。公開照合後mandatory STOP。

- 科学SOURCE／起点commit：`328f31007ee8069d1f7ba24f91029a3748b59109`
- SOURCE tree：`44b024af23b9e907af52b3290f453651e6e99b61`
- 実行branch：`codex/research-c-c1-h4-xdf8-pf-execution-20261026`
- [固定契約](research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json)：scope `C1_H4_SAVED_XDF8_REPRESENTATION_GROUND_PF_V1`
- 契約SHA256：`39af588fee2d879cfa1bf38073dac2bc295cc6f8f610a6be7ef6c105f26fa63c`
- SOURCE manifest SHA256：`4f4d3261411176bebdade53d3a621badfb120839df49e4ac9f59396cad9489cd`
- [機械可読summary](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/execution_summary.json)／[全26セルCSV](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/pf_rounds.csv)／[run01原本・SOURCE・call・STOP](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/run01/)
- [技術準備・再利用・第2/3段階の未採択案](research_c_c1_h4_xdf8_requalification_preparation_20261026.md)

## 固定条件と実行

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36。保存OpenFermion1.6.1 eigh由来λ/Gの**元順prefix8**、number表現、元g補正κ・核scalar・spin/bit/sector順・frame順を保持した。全frame ordered symmetric S2、s=1/4、δ=0.1、m=0..12、PR split非適用。CA=.00159 Ha、DF reserve CA/8=.00019875 Ha、own-XDF8 sector ground oracle、状態準備費用除外。

元P-H4 packet SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。旧公開commit `86b8b002704aced8ca239414080f53793de81658` の固定bytesを新worktree内のignored `.external/P-H4.npz`へコピーした。前段のDF予算科学SOURCE `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`のrank8結果・SOURCE・消費済みSTOPを再利用した。CDF8と旧XDF10の科学的数値評価は行っていない。

独立worktree `/tmp/prt-c1-h4-xdf8-pf-execution-20261026` で、固定SOURCE・契約・入力・既存評価環境のpreflightと新人工22 testsをPASS後、一度実行した。科学SOURCE／runner／契約は無変更。歴史的な`unapproved`ファイル名・metadataを変更せず、承認証明書・原文保存・新grantは作成していない。

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  XDG_CACHE_HOME=/tmp/prt-c1-h4-xdf8-pf-execution-20261026/.cache \
  MPLCONFIGDIR=/tmp/prt-c1-h4-xdf8-pf-execution-20261026/.cache/matplotlib \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h4_xdf8_pf.py \
  --contract docs/research/research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json \
  --input .external/P-H4.npz --execute \
  --output artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/run01
```

**run01は消費済み。上記は実行記録であり、再起動コマンドではない。** retry/resume0、技術中断・修正再実行0。

## 全gate

| 検証 | 結果・数値 |
|---|---|
| SOURCE・入力hash・環境 | PASS。SOURCE21 files、保存入力6 files、元packetを実行前後照合 |
| 保存DF予算 | PASS。保存norm `9.533753411107529e-5`＋allowance `1.480439843662034e-9`＝保存sum `9.533901455091894e-5` Haを引用。新norm計算0 |
| κ・scalar・frame/spin/basis | PASS。元g由来κのmax-entry差 `2.7755575615628914e-17`、核scalar `2.29310124732`をbaseへ1回。元8frame、正しいOF→Qiskit変換、厳密spin lift |
| raw CAR対一般K：base＋8frame | **9/9 PASS**。最大成分差 `8.43769498715119e-15`、固定許容差1e-10、rtol0 |
| raw sum対保存XDF8 matvec target | PASS、max-entry `1.7763568394002505e-15` |
| K sum対同target | PASS、max-entry `9.381384558082573e-15` |
| own-ground residual | PASS、`1.819886133099374e-15` Ha ≤1e-10 |
| 正規化・sector | PASS、正規化差 `5.551115123125783e-16` ≤1e-12、登録sector36 |
| H4 alias | PASS。own-ground energy `−2.166402428462818` Haは[-10,10]内、登録base branch margin `1.6179938779914944` rad >0 |
| S2 s=1／s=4 | **26/26 PASS**。各13roundの誤差が固定sin(π/12)以下 |

CARの完全one-body／square因子がspin sectorを保存することを入力spin liftで確認した。個別Gaussian gateのsector射影で回路を検証した結果ではない。独立一般K係数経路と保存matvec targetを照合し、物理Hamiltonianとの差のDF誤差と表現correctnessを区別した。groundはXDF8固有eigenpairで、旧XDF10の状態・PFセルを流用していない。physical ground overlapは未測定のnull。

## 全26 PFセル

誤差は無次元operator 2-norm `||W_s^(s·2^m)−exp(−i t_m H_XDF8)||₂`、`h=0.1/s`、`t_m=0.1·2^m`。既存PF_rounds／ordered_schedule／expm／eigensector／反復二乗法を無変更再利用した。登録十分条件は全roundで `error≤sin(π/12)=0.25881904510252074`。数値allowance・閾値の事後変更はない。

| m | t_m | s=1 誤差 | s=4 誤差 | 判定（両s） |
|---:|---:|---:|---:|---|
| 0 | 0.1 | 6.2456814960e-5 | 3.9018420889e-6 | PASS |
| 1 | 0.2 | 1.2457699791e-4 | 7.7826633989e-6 | PASS |
| 2 | 0.4 | 2.4648808670e-4 | 1.5398854833e-5 | PASS |
| 3 | 0.8 | 4.7249312197e-4 | 2.9518622524e-5 | PASS |
| 4 | 1.6 | 8.0511676603e-4 | 5.0302112136e-5 | PASS |
| 5 | 3.2 | 9.6812303982e-4 | 6.0490966799e-5 | PASS |
| 6 | 6.4 | 1.2881292529e-3 | 8.0492534026e-5 | PASS |
| 7 | 12.8 | 2.4128589964e-3 | 1.5078343060e-4 | PASS |
| 8 | 25.6 | 4.3039887375e-3 | 2.6895874578e-4 | PASS |
| 9 | 51.2 | 8.5935139144e-3 | 5.3701760379e-4 | PASS |
| 10 | 102.4 | 1.7170157740e-2 | 1.0729911631e-3 | PASS |
| 11 | 204.8 | 3.4278463211e-2 | 2.1421937242e-3 | PASS |
| 12 | 409.6 | **6.8427950728e-2** | **4.2769392104e-3** | PASS |

| 候補 | 最大誤差（m=12） | 登録PF十分条件 | 状態 |
|---|---:|---|---|
| XDF8 S2 s=1 | 0.06842795072754271 | 13/13 PASS | `POTENTIALLY_FEASIBLE` |
| XDF8 S2 s=4 | 0.004276939210418963 | 13/13 PASS | `POTENTIALLY_FEASIBLE` |

NOT_MET／UNRESOLVEDセルは0。決定論なので補正RTE誤差0・attenuation1は構造上の条件である。一方、合成誤差・厳密rounding enclosure・最終CA・shots・総費用はnull／未資格のまま。PF十分条件のPASSをCA達成や費用優位へ昇格しない。

## 科学call・資源・保全

| call分類 | 実績／上限 |
|---|---:|
| saved load／DF gate／sector context／input identity | 4/4 |
| 保存prefix model／36 actual matvec | 37/37 |
| raw9／K model／表現9／sum gate | 20/20 |
| own-ground／alias gate | 2/2 |
| S2 kernel2／PF round26／candidate summary2 | 30/30 |
| **合計** | **93/93** |

journalはattempt93／COMPLETE93／FAIL0／inflight0。新fit／optimizer／新DF分解／DF誤差再計算／CDF8・XDF10の新評価／回路build-compile／合成／CA／shots／総費用／retry／resumeはすべて0。固定Gの内部対角化8件は登録K_modelに含めた表現変換で、新しい二体DF分解ではない。

開始時CPU affinity32、空きRAM約52.5 GiB。既存resource_planで独立s jobを2 workersに分け、task内BLAS1でoversubscriptionを調整した。共有設定変更や恣意的CPU/RAM/wall上限なし。runner計測wall `1.3950881799682975`秒、process CPU `7.525539987`秒、peak RSS `365452` KiB。主にsector36の処理であり、新full256/512計算は行っていない。

[preflight](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/execution_preflight.json)／[postflight](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/execution_postflight.json)でSOURCE21・saved input6＝27件と元packetの不変を確認。run01原本45ファイル（2 kernels、26セル、9 generator／ground／全eigenpair等の保存packet、call journal、STOPを含む）を保全した。旧H2/H4/H6のSOURCE・raw・失敗履歴・grant／ledger／marker／STOP、旧CDF8/XDF10約27%の歴史的結果は変更していない。

## 次に必要な判断・STOP

**XDF8 s=1／4は両方とも、native再資格付けを提案できる候補に残る。** ただし、CDF8とXDF8の同CA総RZ比較は未実施であり、旧XDF10のshots／費用／CAをXDF8へ転用しない。

次は**ユーザー／研究C**が、技術準備報告の「両armへ同じprecision/P0」「full512 Frobenius又は全72入力列のcoverage」「採用するroute」の公平なnative・合成誤差再資格付け条件を判断する。**Codex**は具体的依頼後に最小の次段階SOURCE・人工テスト・契約を準備し、実回路のbuild／compile・数値評価はその契約への別明示許可後だけ実施する。厳密CA保証／状態準備込み資源／無条件優位は引き続き未資格。

今回の固定scopeに残る計算はない。結果・SOURCE・入力・call・STOPをcommit/pushしてremote照合後mandatory STOP。native・合成上界追加計算・CA・shots・総費用へ自動進行しない。
