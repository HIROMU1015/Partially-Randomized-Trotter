# H4 small matrix encoding reuse：run10を継続した独立実装

SOURCE `3beb3870369eeb8ef76b4ead965c886eba7d16e9`、branch `track-a-h4-encoding-speedup-20261010`。
起点REVIEW `95c2269263c92688936343b0423e9f1dfa22281c`、run10 SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
[監査・限定test・binding・差分review](../../artifacts/resource_applicability/track_a_h4_encoding_speedup/2026-10-10/README.md)を一つの資料入口とする。
作業開始は2026-10-10、固定資料は2026-10-11 JST。worktree・証跡はhome配下。

## 変更と理由

driverの受動観測で約100% CPUに対しworkerに待機があり、同じGaussianの小行列のcanonical encodingがproducerの待ち時間になっていた。
[encoding_memo.py](../../src/trottertracks/resource_applicability/h4_geometry/encoding_memo.py)は
`dtype.str / shape / logical C-order raw bytes`をkeyとして、旧encoderで検証した小行列のimmutable JSON textを再利用する。
一つのcandidate内のtrajectoryとcosine/sineだけで共有し、candidate終了で解放する。
4KiB以下・ndim8以下のmatrixのみ対象、cacheは最大1024 entries・保守的に計上したretained bytes 4MiBまでのLRU。
大きなmatrixは従来のLazyList/chunk走査を維持する。

[id encoder](../../src/trottertracks/resource_applicability/h4_geometry/identity.py)のEncodedValueは任意のraw JSONを受け入れず、既存のfinite検査を通した値から作る。
値のsnapshotをkeyとencodingの両方に使うため、配列の変更、signed zero、dtype、shape、stride/orderingを区別する。
phase/control state/label/condition/axisを含む回路全体は従来順序のまま走査・hashし、65536-byte以下のchunk/yieldと独立observerの監視経路を維持する。
IPCは引き続き実QuantumCircuitのpickleで、JSON transportへの変更はない。
Gaussian gate sequence、compiler options、数値精度、資源上限、scientific conditionsは変更していない。

## 限定検証

[計画](../../artifacts/resource_applicability/track_a_h4_encoding_speedup/2026-10-10/test_plan_v1.json)に単一test process、内部thread1、AS1GiB/RSS512MiB、wall60秒、output4MiB、child/affinity/transpile/NPZアクセス0を実行前に明記。
[runner](../../scripts/resource_applicability/run_h4_encoding_speedup_tests.py)と
[tests](../../tests/tracks/resource_applicability/test_h4_encoding_speedup.py)で12件PASS、failure/error/skip0。
wall0.604秒、peak RSS371695616 bytes、証跡3198 bytes。venv・共有環境は変更しない。
初回harnessはdillの`/dev/null` import probeを拒否したため、harnessの許可をそのdeviceだけ補正して実行。初回の科学/compile実行0。

人工4×4 complex parameterを反復した8-system+ancilla回路の1803 instructionsで、既存fingerprint0.30954秒、再利用版0.13509秒（約2.29倍、各1回の限定観測）。
cache596 hit/4 miss、retained7680 bytes。実計算全体の高速化率・72時間以内完了は未測定。
cosine23534 bytes/sine23729 bytesで旧frozen serializerと全bytes/digestが一致。
同じ人工回路のgate counts/depthは変化せず、mutation/nonfinite/symbolic拒否、endian/dtype/shape/stride/signed zero、open control、global phase、Unicode label、condition、cache evictionと容量、paired generator順序を検査した。
人工compile/追加transpile、本番操作、実NPZ読込は0。結果はlocal artifactでありCIやproduction成功を示さない。

## SOURCE・研究範囲と適用

[closure](../../artifacts/resource_applicability/track_a_h4_encoding_speedup/2026-10-10/source_freeze_v1.json)は旧77 pathsのうち3変更/74不変に新module・test・runner3件を加えた80 paths。
旧/new Git blobとbyte SHAをactual SOURCEから固定し、source/testsとREVIEW資料を別commitへ分ける。
同じ回路についてnumerical digestを保持するが、将来新SOURCEを採用する際は既存seed契約のSOURCE_SHAも再結合する必要がある。
run10 worktree・SOURCE・plan/auth・output/control・processを変更せず、そのまま継続する。今回の新SOURCEのlaunch bindingは未seal、approved/runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。

利用者が共有したPR研究の公開起点は`track-a-ax2b-h4-post-review-20261010` / `a87cf25548a3b93262027ce780317872d7c4e883`。
run10は別途認可されたH4 six-geometry signal/compile mapであり、AX2b accuracy/u/NG certificationやH6承認の証拠に読み替えない。
本変更はencodingの実装・人工検証であり、新たな科学的結論やcompiled-cost結果ではない。

独立担当の変更箇所だけのreviewはPASS、具体的P1/P2なし。追加tests/compile/入力読込/本番操作0。cacheはdriverの単一producer用でありthread間共有を想定しない。
