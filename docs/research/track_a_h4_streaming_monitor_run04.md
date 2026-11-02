# H4 run04：exact fingerprint分割・監視維持・12 worker再実行

利用者の「その修正を入れて、ワーカー数を増やして再実行して」と、停止後の「続きを行って」に基づく。
local実装検査はPASS。科学的成功・全campaign完了は未確認。起動直前fresh gatesがPASSの場合だけ、
新run04を一度起動しMAP_COMPLETE_STOPで停止する。自動retry/resumeはしない。

## 停止の証拠と修正

run03はfresh preflight PASS後12 owned workersを起動し、候補間に3 invocationsを投入した後
monitor STOPで終了した。compile完成0、signal0。driver/全worker終了をread-only確認した。
旧source・run01/run02/run03・全input/partial records/journals/logsを削除・移設・上書きしない。
旧監視の元の例外は保存されなかったため、run03そのものの正確な停止原因は未確定である。

人工JSONメタデータ（256×256のcomplex-hex parameter metadataを128回繰返し、科学array/回路0）で、
旧fingerprintの一括json.dumpsが監視threadを妨げ、monitor interval/freshnessを再現した。
旧観測のsince_previous_checkは6.6886 s、5秒上限を超えた。JSON処理のGIL占有が
run03の不規則な監視間隔と整合する有力な原因であり、実runの唯一の原因を証明したとはしない。

fingerprintは同じexact変換・sort keys・UTF-8・separators・nonfinite禁止を維持し、
JSONEncoder.iterencodeでencodingを分割して64 KiB単位でSHA256へ渡し、own process内でsleep(0)する。
同じ大きいpayloadのdigestは旧/新とも8bc4b0421dd538a1c9078229e66ea184da5de5cc57b6c45a6485f9ffcf9cf216。
新方式の診断は全監視PASSでthreadも終了した。この診断のelapsedは旧29.42s/新59.60sで、
新方式を高速化結果とは主張しない。巨大な単一encoded stringの追加allocationも避ける。
65直接関連testsはPASS（failures/errors/skips0）。追加transpile・分子/科学array・production workersは0。
signed zero、complex、極端finite float、Unicode、ordering、unsupported/nonfinite STOP、
旧canonical SHAとの一致、候補間queue、seed/axis/metrics、台帳/認可、監視原因ログを検査した。

監視5秒、driver/worker AS/RSS8 GiB、headroom16 GiB、PSI0/OOM増加0を緩和しない。
watcher停止時は最初の例外内容をbounded driver logへ出し、cleanup中の例外に隠さない。
候補間compile queue、最大12 outstanding invocations、logical集計順、cache scopeはrun03修正を維持する。
compiler options/45 dependencies・科学input/math/circuit条件・seed算法/masterは変更しない。
SOURCE_COMMITがidentity/seedへ入る従来仕様に従い、新sourceへ再結合した結果だけを新runへ保存する。
旧random/partial metricsを新sourceの結果へ混合しない。

## 固定条件・入力・累積予算

- SOURCE_COMMIT：461ed3e5b77ebb897856a5c88382d035cbb2d668
- branch：track-a-h4-streaming-monitor-run04-20261006
- actual checkout：/tmp/track-a-h4-streaming-monitor-run04-20261006。/homeのoutput空きを圧迫しない別filesystemの独立worktree。共有設定変更なし。
- plan SHA：0e67ef9a81f201e3e0939a45b497b7bf0e2138d1e41b48d383444df8c9d8c404
- plan fingerprint：8c79bb8433a72e47f13a7dc10544cd823e18d43fe85282bb9fa4b77ce58cb074
- auth digest：ad6951a44ad71d7b6ae8e7db1ba5184b4539134284d627771073f82d39d8707b
- review digest：761a165821b06d9339ae8dc5b0ec43381f49266270a8291f7a34f58a6bf71e3d
- CPU [3,5,6,7,8,9,10,11,12,13,14,15]、12 owned workers、own新規driverだけmask0xffe8。
- fresh memory>=120 GiB、quota確認済みnonroot disk>=3.5 GiB、available inode>=560000。
- run02の凍結6入力をread-only再利用、生成source049e69919af16ad29a67a217dc7a407d6b1754a6・freeze・32 arrays identitiesを保持。
- old charge165168060 bytes（run02を含むrun03 journal）をcarryし、新journal128-byte carry rowも課金する。
- old invocations7件を含めactual cap74784、new attempt残上限74777。
- prior wall5015.033646528935sをcarryし72h累積上限。run03最終failure logまでの経過+60sを保守的に含む。
- 全campaign10 GiB charge cap。入力再生成なし。stage容量planningは既存12 worker小計3576795136 bytesを維持する。

H4 linear neutral singlet/STO-3G、4 spatial/8 system+ancilla1、DF12。
6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、T=0.8、二次DF-prefix PF/canonical finite-RTE。
218 templates/距離、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1。
固定r/K、random32 paired trajectories/master20261006。signal1308・logical wrappers74784、
H6/Track B/追加trajectory/anchor/長RPE/最終総costへ進まない。

## launch直前と記録

source19/45 dependencies/compiler・actual checkout/plan/auth/review・old freeze・6 NPZ streaming SHAをfresh照合する。
run02/03 journalsの不変性と累積charge・invocation/wall、旧own PID/worker全終了、新output不在を再確認する。
CPU online/cpuset/12 distinct cores/NUMA0・3秒受動SMT load・全祖先CPU quota、memory/pressure/OOM、
fixed output mount/user-group-project quota/available bytes/inodesを確認する。不合格・欠測なら科学runnerを呼ばずSTOP。
承認reviewは実際の利用者指示の転記であり、外部reviewerの承認を捏造しない。旧false草案を保持する。

source/plan/auth/reviewは本bundleに固定する。runnerはこのcheckoutの
scripts/resource_applicability/run_h4_geometry_signal_compile.py。
control入口：/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/h4-signal-compile-run04-streaming-monitor/README.md。
実起動argv/時刻/PID・fresh preflight・progress・最初のfailure reasonはcontrolの別file/logへ記録する。
科学NPZ・production runtime/checkpoint/cacheはcommitしない。

[bundle入口](../../artifacts/resource_applicability/track_a_h4_streaming_monitor_run04/2026-10-06/README.md)。


## run04結果とrun05の修正

run04は5compileを投入後monitor interval/freshness STOPとなり完成record/signal0。全own process終了、旧全証跡保持。巨大exact-tree変換中の停止を[run05のlazy変換と入力/予算binding](track_a_h4_lazy_identity_run05.md)で補修する。
