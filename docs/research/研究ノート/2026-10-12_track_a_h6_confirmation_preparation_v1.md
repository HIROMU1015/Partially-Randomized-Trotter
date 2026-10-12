# H6独立32 trajectory確認：実行準備契約 v1

状態：**PREPARED_NOT_AUTHORIZED**。候補固定と技術sealは完了、実確認計算0、新approved grant0。`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`。本計算は別の利用者明示承認・一回grant・fresh prelaunchを必要とする。

[一つの資料入口](../../../artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12/README.md)。

## 起点・科学条件・保全

起点branch `track-a-h6-roundoff-resume01-20261012`、結果commit `df99d6ee28942df8d12aac3c4db7cd202df55d77`。origin branchのSHA一致を確認した。元探索SOURCE `78a39051d8b6d41c33d592ecfe172f8824adfaeb`、科学SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`。新確認SOURCE `73018cbcc39803890974c7ee51401063e6c8daf6`。

linear H6・1.00Å・STO-3G・12 modes・alpha3/beta3 sector400・保存state・T=0.8。採用signed generation-order DF rank19/tol-only1e-8/cutoff0/weighted Hermitian projectionを維持する。全92候補の科学条件はprefix5/10/15/full19、q1/2/4/8、delta0.8/0.4/0.2/0.1、B2 r1/2/4・R=qr・K2/4の元契約どおり。新DF/state/input/reference/signalを生成しない。

経験的numerical allowance、normalization、shot会計、finite law、Random seed namespace、RZ/CX/depth/size定義、full measured Hadamard wrapper・state preparation除外を変更しない。compilerはQiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routing指定なし。

原科学224・原実行closure240をbytes/Git blobで保全し、新runtime5 fileを加え245 closureへ固定した。旧449＋server Phase A447＋保存7＋新135の592/592 group・1188 wrapperを再受理する。partialや旧pilotを平均へ混ぜない。既存H4/worktree/venv/共有環境/他jobは変更しない。

## 選抜規則と固定候補

元規則「各epsilonのB2 shot込みpoint G_RZ上位2、tieはID辞書順、和集合最大8」をそのまま適用した。全592完了gateを通してから選抜し、探索rows digestと全group path/SHA・source・task・seed・両軸の来歴をbindingした。

| epsilon_sig | 第1候補 | 第2候補 |
|---|---|---|
| 0.05 | `B2_p5_q1_2nd_r4_K4` | `B2_p5_q1_2nd_r4_K2` |
| 0.01 | `B2_p5_q2_2nd_r4_K4` | `B2_p5_q2_2nd_r4_K2` |
| 0.005 | `B2_p5_q4_2nd_r4_K4` | `B2_p5_q4_2nd_r4_K2` |
| 0.001 | `B2_p5_q8_2nd_r2_K4` | `B2_p5_q8_2nd_r2_K2` |

和集合は上表の8構成。すべてprefix5で、q1/q2/q4はr4・K2/K4、q8はr2・K2/K4。これは結果前の確認用集合を固定したもので、population winnerの認定ではない。

各構成32 fresh whole trajectory、replica0–31、計256 group・cosine/sine計512 wrapper。seedは原式 `SHA256(H6_MATCHED_V1_FRESH_COST/{cell_id}/confirmation/{replica})` の先頭8 byte big-endian。256 seedの一意性と全探索seedとの非重複を検査した。X/Yは同じevent列を共有し、独立標本数は32で64ではない。

探索8と確認32をpoolしない。選抜済みB2だけを新取得し、B0/B1の保存deterministic費用は再利用する。64未確認B2をcoverageへ明記し、確認結果で候補・標本数を増やさない。追加探索・H8・自動retry/resume・次stageは認可しない。

## 希少Taylor次数と判断の限界

finite lawの保存確率から、固定32中のnonzero order未観測確率と、探索で未観測だったorderが確認32に現れる確率を事前記録した。新trajectoryをsamplingして確率を作っていない。

| q / r / K | 確認32すべてでnonzero orderなし | 探索で未観測のorderが確認32に出る確率 |
|---|---:|---:|
| 1 / 4 / 2 | 12.131% | 87.869% |
| 1 / 4 / 4 | 12.060% | 0.58553% |
| 2 / 4 / 2 | 34.200% | 0% |
| 2 / 4 / 4 | 34.175% | 65.825% |
| 4 / 4 / 2 | 58.344% | 0% |
| 4 / 4 / 4 | 58.339% | 41.661% |
| 8 / 2 / 2 | 58.344% | 41.656% |
| 8 / 2 / 4 | 58.339% | 0.0094582% |

32標本でもnonzero未観測の可能性が約12–58%ある。確認で得たpaired mean/SD/SEと、observed order count・既知の未観測mass・all-zero確率を別々に保存する。sample SEが小さい/0でもpopulation tailを保証しない。`population_cost_tail_bound=null`、`UNDETERMINED_POPULATION_MEAN`を許す。instruction guardは母平均費用上界ではない。

この限界を理由に、結果後の追加sample、order条件付け、層化への変更、K/R/grid変更を行わない。必要な新研究方針は結果をGPT/利用者へ戻して判断する。経験的u・u10倍感度・G(P)の共通state-preparation感度は既存会計どおり出力するが、数値certificateやchemical accuracyへ読み替えない。

## 累積budgetと最小変更提案

過去の既知callと独立予約を保全し、失敗分返却・budget resetなし。未知の過去actual totalはnullで、予約を実消費と呼ばない。

| 項目 | known carry | 旧未完了分の独立予約 | 新256のdemand | admission合計 | 原上限 | 提案上限 |
|---|---:|---:|---:|---:|---:|---:|
| compile | 1188 | 12 | 512 | 1712 | 1712 | 1712 |
| control_probe | 360 | 4 | 0 | 364 | 1024 | 1024 |
| trajectory | 578 | 6 | 256 | 840 | 832 | 840 |
| occurrence | 2176 | 48 | 960 | 3184 | 6656 | 6656 |

**trajectory累積上限832→840だけが最小の変更案**。元科学標本数（探索576・確認256）は変えず、既知diagnostic2と停止分予約6を上限へ加えるための提案である。`cap_amendment_approved=false`を維持し、新grantはこの変更の明示承認を必須にする。compileは予約込み1712で原上限と一致し、余分なcompile campaignやretryを入れない。

known primitive1470・reference_matvec400、旧未知actual/STOP/carryは原記録のまま。本会計は既知値＋保守的予約のadmissionであり、未記録の全歴史をexact totalとして復元したものではない。

## 環境・CPU・memory・監視

既存private venv `/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv` を変更せず使用する。Python3.11.0rc1/Qiskit1.3.0/NumPy1.26.4/SciPy1.14.1/Numba0.60.0/OpenFermion1.6.1/PySCF2.7.0、native Nature0.7.2/Algorithms0.3.1/Aer0.15.1/psutil5.9.8を固定profileに結ぶ。profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`。

利用者承認済みroundoff bridge・旧source/params/order/sign/seedを保全する。旧host/compilerとのoutput bit完全同一性は未検証のまま。丸め差の許容を、seed/ordering/compiler/回路取り違えの無視には拡張しない。

今回の優先作業は独立確認準備。保留中のhash再利用・8worker改修は混ぜず、実績のある4workerのsource・資源設定を基準にした。

| role/上限 | 固定提案 |
|---|---|
| worker | 4、CPU20/21/24/25、各内部thread1 |
| coordinator | CPU22/23、AS soft2/hard32GiB、RSS2GiB |
| worker AS / sampled RSS | 各32GiB / 9GiB |
| aggregate RSS / job RAM予算 | 40GiB / 64GiB、kernel RAM予約なし |
| phase/total wall cap | null / null |
| monitor | interval1秒、観測/staleness最大5秒、journal10秒emit・32MiB |
| pressure/OOM | 元host full PSI>=5%・cgroup full PSI>0・新OOMでSTOP |
| launch headroom | 元fresh host216.25GiB/effective64GiB、running168.25/16GiB |

own affinity/nice10・Python -I -P -B・Qiskit num_processes1/parallel falseを維持する。新role/GPU/taskset/共有scheduler設定を追加しない。凍結monitorをphase directory名だけ変更して再利用し、PF_EXITING限定1秒退出待ち、欠測RSS保持、所有PID/startticks/group/session、ESRCH扱い、最初のSTOP理由保持、全wait/reap/pipe/log閉鎖を維持した。

準備時読取snapshotは利用可能RAM約465.01GiB、disk約401.82GiB、inode約2.20億、quota KNOWN。affinity0–63、提案CPUは利用可能で物理coreが重複しない。これは準備時観測で、launch資源予約・直前のPASSではない。

## output・一時cache・容量

新exclusive outputは `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-confirmation-preparation-20261012/artifacts/resource_applicability/track_a_h6_independent_confirmation_v1/2026-10-12/launch_confirmation_v1`、未使用。新one-shot controlは `.server-preparation/h6-independent-confirmation-v1/one-shot.lock`、未作成。準備・test/log・cache・一時fileはすべてhome配下。

保存した各選択構成8 groupの最大bytes/block/inodeを32へ比例した参考値はraw91.38MiB、block込み約106.88MiB・file inode5248。journal32MiB/log1MiB/一時cache提案256MiB/report16MiBを加え約411.88MiB、inode余裕2万を計画する。希少orderによる上振れの実証上界ではない。

元global output2GiB・worker writer32MiB・log合計1MiB・FS必要空き10GiB/10万inode・quota検査を保持する。256 workerすべての32MiB ceilingの和は8GiBでglobal capを超えるため、全ceiling同時確保や完走保証を主張しない。journal/cache/metadata/partialもaggregateに算入し、超過時fail-closed STOPとown cleanupを行う。

## 実装・限定検証・起動gate

新moduleは `h6_confirmation_contract_v1.py` と `h6_confirmation_execution_v1.py`。metadata preparer・read-only auditor・default dry-run runnerを追加した。科学portは元 `h6_roundoff_cost_port_v1.cost_task` だけで、独立seedの確認taskへ接続する。

69 local限定test合格：元top2/辞書順tie、32独立seed・pool拒否、carry/予約/832→840最小案、旧/draft/tampered grant拒否、phase/output/lock/source/CPU、256 group完了条件、元monitor/workerとの最小差分parity、dry-run science importなし、最大4 tiny childの所有・STOP・cleanup。driver AS2GiB/thread1/wall120秒・home-only tmp、tiny child各AS128MiB。追加transpile・実H6 array decode・新signal/trajectory/compileは0。

元結果CSVのinteger gate数とshotsから別途top2を再構成し、256 seedをliteral SHA式で再導出して一致。source/profile/input byte identity、全592保存group、archive/receipt/carry、未使用output/controlを再監査した。これはlocal metadata/算術検査であり、独立scientific review・CI・実32確認PASSではない。

launchには新 `h6_confirmation_authorization_v1` が必要。SOURCE/profile、全探索binding、固定selection/task digest、840 cap案、resources、output、one-shotをdigestへ結ぶ。旧grantの再利用・draft・source/seed/cap/roleの変更は拒否する。準備manifestのflagsは未承認のまま、未来のexact grantだけが一回を認可する。

明示承認後も同じforeground launchでsource/profile/input/selection/carryとCPU/memory/PSI/OOM/FS/inode/quota、未使用output/lockを直前に再検査する。完了またはfail-closed STOP後に終了し、retry/次stageへ進まない。

## 公開・次の承認事項

専用branch `track-a-h6-confirmation-preparation-20261012`。新SOURCEとREVIEW_BUNDLEを分け、source/tests/schema・軽量準備資料だけを明示stage/commitしてoriginへnon-force pushし、remote SHAを照合する。NPZ・実runtime/checkpoint/cache・credential・内部SSH情報は追加commitしない。旧worktree/成果/dirty/untrackedは保持する。

次の承認案は「固定8構成それぞれ独立32 trajectory（256 group/512 wrapper）を、SOURCEと4worker資源に結んだ新grantで一度実行し、完了またはSTOP後に停止。trajectory累積上限832→840のみ承認、その他科学条件・compiler・上限・carryは維持。自動retry/追加sample/次stage/H8なし」。

本段階ではgrant発行・実32計算を行わず、公開確認後にSTOPする。hash再利用・8workerの別依頼は保留のまま。
