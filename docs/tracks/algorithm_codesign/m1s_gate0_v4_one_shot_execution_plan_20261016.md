# M1-S Gate 0 v4 初回one-shot実施計画（2026-10-16）

ユーザーはGate 0初回one-shotを具体的に提案した直後、別turnで「次の作業に進んで大丈夫」と実施を承認した。
正式採択と今回の実施承認を分離し、approval basisの原文/contextを新規保存する。IDE fieldsは根拠にしない。

正式契約commit5e57955baa79e42f37aa233ac011c60539e431fd、SOURCE606036db52da2f080cf846e48824e5e418430762を保持。
H4 linear1.00 Å/STO-3G/8 spin orbitals/rank12・保存順G3/G6/G9・非冗長Pauli-tail・保存数値oracle、delta windowなし。
17精度項目・1e-8 Ha+B_res・CONDITIONAL_ENGINEERING_NOT_CERTIFIED・runtime/resource・one-shot/STOP条件を変更しない。

事前metadata照合ではSOURCE7、入力8652bytes、正式契約hash、Python3.11.0rc1、NumPy配布物1.26.4と6build hashが一致。
実import、実際のresource limit、数値correctnessは認可された初回run内で確認する。事前の科学計算/NPZ配列decode/formula評価は0。
CPU/RAM/GPU inventoryをmetadata_preflight.jsonへ保存。採択どおりCPU1 worker/各thread1/GPUなし、4 GiB AS、result/log各256 MiB。
sourceのcached JW bilinears、全split共有DF block map/reference、bounded mask cachesを再利用。
主負荷はPauli aggregationと256次元reference products。過去Gate0 timingなし、追加benchmark/source改変/speedup主張なし。

別のauthorization-only commit Aを作り、source<正式契約<Aを固定してpushする。
new result namespaceを一つだけ登録し、clean dedicated HEAD=Aから既存v4 runnerを一回だけlaunchする。
外側stdlib launcherは/tmpにA固有のexclusive immutable launch markerを作り、worker/thread条件を固定してfrozen runnerへexecする。
このmarkerはguard/runtime失敗時も消去せず、科学処理に到達しなくても自動再実行しない。SOURCEのexclusive result directory guardも維持する。

初回数値correctness・input Hermiticity・reference/JW再構成・state/residual・精度予算gateを先行し、合格しprecision-resolvableな分割だけW/v/rhoを評価。
全G3/G6/G9行をstatus付きで保存する。TECHNICAL_STOP/PRECISION_UNRESOLVEDを科学no-go・rho=1へ読み替えない。
閾値緩和・再生成・simulation/compile・T1c/Q4/Q5・scope拡張・自動retryなし。
scientific wall-time capなし。tool pollingは実験capと分離し、実行中はprocess/出力状況を観測する。
run後に結果・context・log・launch marker・実行statusを保全し、read-only identity/status監査と公開を行いmandatory STOP。
