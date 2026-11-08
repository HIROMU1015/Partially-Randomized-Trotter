# 通常CDF＋Z再配置＋PR：一回限りの科学実行・環境例外STOP

2026-11-08 JST。**CASE C：環境例外により主比較未完了。通常CDFの条件付き総RZ・選択r・統計判定は未取得（null）。**

ユーザーの明示承認に基づき、固定`execute(root)`を一度だけ呼び出した。最初のhead shell検査でQiskit AerのCUDA共有ライブラリのシンボル解決に失敗し、既存one-shot saverが`STOP.failed=true`を保存した。再配置・PF/RTE・RPE・native価格・final・ENTRY主比較へは到達していない。retry/resume、SOURCE変更、環境修復後の再起動は行わない。

これは通常CDF構成が科学的に不適格という結果でも、ENTRYの勝利でもない。保存ENTRYの総RZだけを使って差・削減率を作らない。

## 1. 固定SOURCE・候補・実行条件

| 項目 | 今回使用した固定条件 |
|---|---|
| 科学SOURCE commit | `bde646c2b0d275c66d9890eee835d4692f55cafb` |
| SOURCE manifest | `artifacts/research_c_cdf0_relocated/2026-11-08/technical/SOURCE.json`、112ファイル |
| SOURCE SHA256 | `640052dfeed3b886bb95e11abb410e3275cdda1ffe82661e544037c120f94c32` |
| proposal | [固定proposal](research_c_cdf0_relocated_contract_proposal_20261108.json)、SHA256 `c24cfd9d11940954d452379f7c4c8a6f397ee289611fb7d7038ad1349df9c9a4` |
| 候補 | `CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8`、Stage A V2のγ=0最終number-CDF8 |
| 保存J/U SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| 元h/g/constant input identity | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 物理系 | Linear H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36/full256/control512 |
| 分割・予定再配置 | 元順F0〜F7、LD2、F2〜F7の線形Zを保存κへ、対応scalar shift一回、frame I/ZZを保持 |
| 予定PR/PF | ordered partial S2、s1、δ=h=.1、K0、r={8,16,32,64,128}、m0..12 |
| state・精度 | own-DF ground oracle、prep除外、CA=.00159 Ha、DF reserve CA/8、βtotal=π/6、βstat=π/12、failure=.01、axis=.01/26 |
| compiler・検証 | Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、P0 phase保持、fenced serial、full512/実逆/X-Y |
| selection/final予定 | 各適格rのselection32だけで固定、選択1 pairの独立final64。保存ENTRY E2/r16 final64と一主比較 |

準備時の`scientific_execution_permission=false`は固定proposalの歴史的フィールドとして維持した。今回の会話の実行指示を許可として扱い、承認証明・原文保存・grant等は作成していない。[準備報告](research_c_cdf0_z_relocation_comparison_readiness_20261108.md)は前段の技術記録として保持する。

元の準備worktreeを保全し、`/tmp/research-c-cdf0-z-science-20261108`、branch `codex/research-c-cdf0-z-relocation-comparison-execution-20261108`を同commitから作成した。正式運用規則2026-10-25の§8.1/8.2/8.4/10/10.3を確認し、新しい科学runnerやpipelineを追加せず既存関数だけを起動した。

## 2. 最終preflightと一度だけの起動

既存の受動preflightはPASS。SOURCE112、保存input32、B1契約による候補/reference/ground来歴、保存DF適格metadata、κ/scalar identity、package metadata、compiler profile、fresh run01不在を確認した。元SOURCE68も固定hashと一致した。[preflight原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution/preflight.json)。

affinity128 CPU、load約.46/.41/.38、MemAvailable約1,024,336,052 KiB、専用worktree作成後disk空き594,290,245,632 bytesを観測した。CPU/memory pressure avg10は0、I/O some avg10は1.90%。cgroup v2のcpu.max/memory.maxはこの位置から参照不能で、quota無制限と認定していない。共有設定・package・他processは変更せず、新しいCPU/RAM/worker capも追加しなかった。

以下は**実行履歴であり、再実行の許可ではない**。

```bash
/usr/bin/time -v -o artifacts/research_c_cdf0_relocated/2026-11-08/execution/process_time.txt \
  env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/tmp/research-c-cdf0-preparation-deps-20261108:src \
  /home/AbeHiromu/projects/partially-randomized-trotter/.venv/bin/python -B -u -c \
  'from pathlib import Path; from trottertracks.c_cdf0_relocated import execute; print(execute(Path.cwd()))'
```

stdout/stderrは[console.log](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution/console.log)へ保存。`execute()`呼出し1回、GNU timeのwall .63秒、user1.80秒/system6.36秒、peak RSS111,616 KiB、exit1。[time原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution/process_time.txt)と[終了観測](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution/completion.json)を保全する。これは失敗した起動の時間で、科学計算全体の所要時間・高速化性能ではない。

## 3. 失敗段階と静的に確認できる原因

`c_cdf0_relocated.evaluate`はκ/constant、保存J/U、保存B1のgenerators/Hを読み込み、headのraw/compiled QPYをdeserializeした。その後の最初の`shell_gate`内で、`shell_assessment → decode → c_full_unitary.decode → import qiskit_aer`が失敗した。

例外は、venv内の`nvidia/cusparse/lib/libcusparse.so.12`が要求する`__nvJitLinkGetErrorLogSize_12_9`（version `libnvJitLink.so.12`）を解決できない、という`ImportError`である。完全なpathとtracebackは原本log/STOPを参照する。実際に選択されたnvJitLinkのpath/versionや、異常の発生時期までは今回特定していない。

固定decoderは`AerSimulator(method='unitary', device='CPU', precision='double')`を使うが、そのCPU経路を作る前のAer importが共有ライブラリ解決を要求した。GPU科学計算を試みたという意味ではない。代替decoderへの切替、部分空間/probeへの縮小、閾値緩和は行わない。

**package版のmetadata一致は実行時ABI/linkの成立を確認しない。** 今回の最終preflightはmetadataとbytesの検査で、このAer import/linkを検査していなかった。前回の人工33 PASSはその時点・人工範囲の証拠として保持するが、今回環境のAer起動資格や実H4 correctnessへ格上げしない。

科学開始後なので、固定バージョンの専用環境への修復であっても、このrun01の再開・再起動はしない。SOURCEもproposalも変更せず、例外を保存したところで科学処理を停止した。

## 4. 結果・比較の取得状態

| 要求された結果 | 状態 |
|---|---|
| 保存DF精度 | 再利用可能な既存数値pilot。sector norm=`7.018855038843621e-9 Ha`、guard=`3.278781198621627e-9 Ha`、保存和=`1.0297636237465247e-8 Ha`、CA/8予算PASS。新DF normは0 |
| J/U・κ/scalar入力 | 固定ファイルのloadとκ/scalar array identity照合まで。新fitは0 |
| 再配置・full-H同値 | NOT_RUN。保持すべき代数規約は固定したが、今回の実分子gateは未実施 |
| 再配置後tail mass | null。中立C0や旧候補のmassで埋めない |
| r8/16/32/64/128適格性・PF/RTE | 全てNOT_RUN。全r不適格と判定したわけではない |
| attenuation・shots・W | null／NOT_RUN |
| native RZ/CX/depth | null。新kernel build/compile/実逆構築は0 |
| selection r・final64 | 未実施、selected_r=null、final総RZ=null |
| 保存ENTRY E2/r16 | 元の独立final64結果75,621,395,471 RZを参照。新ENTRY評価0 |
| CDF0−ENTRY差・削減率・名目95%区間 | 全てnull、主比較NOT_RUN。点値・区間・勝敗を推測しない |
| correctness・精度資格 | 今回の実H4 full-H/full512/phase/inverse/X-Yは未検証。科学的な候補不適格は未確定 |

上の保存DF数値の条件はH4/1Å/STO-3G/number-CDF8/元順LD2のStage A γ=0終端、DF評価にPF時間窓は非適用。保存ENTRY費用は同じ物理系、CDF8/LD2/δ=.1/s1/K0/r16/m0..12、own-ground・prep除外・P0/fenced serial・ideal finite-RTE/exact synthesis仮定の有限価格proxy。今回の失敗で、その既存値の資格や過去判定を変更しない。

## 5. 実行範囲・登録上限との照合

| 範囲 | 今回 | 登録上限 |
|---|---:|---:|
| execute呼出し | 1 | 1 |
| CDF候補load | 1 | 1 |
| Z再配置／PF kernel／exact evolution | 0／0／0 | 1／1／13 |
| PR reference round | 0 | 65 |
| selection native／final native | 0／0 | 160／64 |
| kernel build／compile／実逆build | 0／0／0 | 224／224／448 |
| native kernel full512 decode | 0 | 896 |
| shell full512 decode | 最初の1件を試みてimport失敗、完了0 | 完了6 |
| 主比較 | 0 | 1 |
| 新fit／DF norm／ENTRY／強D／量子shot | 全て0 | 全て0 |
| retry／resume | 0／0 | 0／0 |

親callbackは、固定された逐次SOURCEと保存tracebackから`kappa_load`/`candidate_load`/`reference_load`各1 COMPLETE、最初のhead `shell_gate`1 FAIL、計4 attempts（3 COMPLETE＋1 FAIL）と復元できる。合計6配列のload、2 QPYのdeserialize、最初のoperator decodeへの到達に対応する。これは呼出し境界の集計で、成功した科学評価4件を意味しない。

失敗した既存runnerは`call_counts`を結果へ保存する前に例外終了した。従って上記を独立した実行時per-call ledgerが存在するかのようには表示しない。[監査JSON](../../artifacts/research_c_cdf0_relocated/2026-11-08/execution/result_audit.json)にもこの証拠の限界を明記する。登録外の追加評価・比較は0。

## 6. 原本保全・公開範囲

[run01](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run01/)にはSOURCE.json、contract.json、failure.json、STOP.jsonの4ファイル、28,328 bytesがある。すべて直接公開する。新しい科学raw binaryやnative標本は生成されていない。実行log/time/preflight/completion/監査も公開し、巨大rawを非公開保存したという扱いにはしない。

[STOP原本](../../artifacts/research_c_cdf0_relocated/2026-11-08/science/run01/STOP.json)はfailed=true、completed=[]、retry/resume0、mandatory_stop=true。失敗を正常完了へ書き換えない。run01を消さず、未実行の標本数や費用を0という科学結果へ変換しない。

postflightでSOURCE112・保存input32・旧科学SOURCE68のhash一致、固定proposalとrun内contractのbyte一致、SOURCE manifestのbyte一致を確認した。旧raw・SOURCE・契約・grant/ledger/marker/STOP、元mainと準備worktree、共有venv・他processは変更しない。今回の科学SOURCE変更は0。

## 7. 残る判断とMANDATORY STOP

主目的の通常CDF総RZ取得・ENTRYとの比較は未完了。結果分類は**CASE C：環境例外で未解決**。これは科学的な不適格CASE Dや手法間の負結果ではない。

次に必要になり得るのは、固定Aer0.15.1とその共有ライブラリ依存を専用ユーザー環境で整合させ、import/linkまでを科学実行前に確認する技術作業である。今回のSOURCE自体に修正が必要かは、この例外からは決めない。修復後に科学計算を望む場合も、消費済みrun01ではなく、別途明示承認されたfresh実行範囲が必要になる。

今回は環境修復、別namespaceの再試行、新fit、ZZ-only正則化、r/K拡張、強D再計算、H6、重要科学レビューを開始しない。失敗原本・監査・本報告をcommit/公開し、remote SHA/tree・本文・SOURCE/result整合・worktree cleanを照合した後、**MANDATORY STOP**する。
