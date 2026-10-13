# H4 B0+I：検証用実装・人工tests・6 compile準備

2026-10-13。**PREPARATION ONLY、実H4の新compile/signal/trajectory0。**
B0+Iの正式baseline採用、比較集合、主要claim、原稿v0.3は変更していない。
新branch `track-a-h4-b0i-compile-preparation-20261013` は統合commit
`f6b0cb5e2ddabd2d7919d0822c58324c7338bac0` から独立し、旧worktree/raw/checkpoint/ledgerを保全する。

SOURCEは `8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b`。
[新bundleと全binding](../../artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/README.md)、
[実行・STOP契約](../../artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/EXECUTION_CONTRACT.md)、
[予算と6/10/240比較](../../artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/BUDGET_AND_SCOPE.md)を参照する。
元科学SOURCE `d0726e946f89f8c30a08d383fb8ff48b363629ec` とseed/input SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525` は履歴とidentityとして維持する。

## 実装差分

新しい小さなadapter `h4_b0i/adapter.py` は独立method `B0_PLUS_I_VALIDATION`、
template `B0I-validation-p{prefix}-q{q}`、namespace `h4-b0i-pilot6-validation-20261013` を持つ。
既存B0 preparationをコピーし、**nuclear constantへdiscarded-tail Iだけを加算**する。
保持済みone-body/DFのI/Z/ZZ係数、順序、basis、PF S2を変えない。
追加済みmarkerによりadapterの二重適用を拒否し、元prep/source/candidate IDを変更しない。

既存immutable `circuits.build_evolution` の `p(-delta*constant,ancilla)` の角度へ統合し、
不要な新gateをappendしない。total追加phaseは `-q*delta*c_tail=-T*c_tail`。
controlled low branchはidentityのまま、高branchは元B0に `exp(-iT*c_tail)` が掛かる。
cosine H-E-Hとsine H-E-Sdg-Hを別検査し、state registerのunitaryとbranch relative phaseを分けた。
このphaseはfull wrapper全体のglobal phaseとして捨てられない。

tail Iはsourceの `lambda/4*((Tr G_L)^2+Tr(G_L^2))`、lower-Hermitian規約。
人工fixtureはupperを異ならせてもlowerの定義を保ち、signed lambda/zero/nonfiniteを検査する。
実H4では先行診断の保存scalarとsignalを保持し、新eigh/signal simulationを準備段階で行わない。
承認後も実signalやtail eigensystemを計算せず、compile lowering用のkept blocksだけを既存builderへ渡す。

numeric canonical encoderは元sourceを再利用する。phase/ordering/global phase/control/axisの数値FPと、
新SOURCE/input/DF/state/template/control/phase/namespaceを含むvalidation FPを分ける。
zero tailではnumeric circuitが元B0と同一でも、新method/source/task identityは独立。
compiled RZ増減は未測定であり、pre-transpile命令数の一致や数学的同unitaryからC_I=C_B0を推論しない。
旧B0 metric/cacheを新しい実測として再利用する経路はない。

## 人工/mock検証

44 tests PASS、fail/error/skip0。1 process、内部thread1、AS2GiB、CPU120秒、wall180秒。
実/人工transpile、process spawn、NPZ/NPY読込、molecular/CuPy importをguardで禁止した。
人工回路は2-system＋ancillaの最大3qubitで、q1/2/4/8、正/負/zero scalar、非可換basis/kept blocksを含む。

- lower trace、tail-only・二重加算防止、元prep/source bytesの不変性。
- low/high branchのunitary関係、scalar符号、総位相、phase位置、pre-transpile命令系列、cos/sine期待値、global phaseとの区別。
- source/input/template/control/axis/signed zero/nonfiniteのFP、保存3構成のtarget/biasとの算術一致。
- 新schema、6 registry、旧grant拒否、未承認拒否、CPU明示許可、plan/auth/review tamper、retry/scope/carry reset拒否。
- 7件目/unknown/duplicate拒否、reservation I/O失敗、mock compiler両軸とinput mutation拒否、complete-only metric/source/FP。
- partial pipe deadline/監視継続、pressure/memory/CPU load/physical core/block/inode/quota欠測、host OOM増加。
- ESRCH継続、非ESRCH不合格・後続wait/reap、所有不一致でsignalなし、二重cleanup、最初のSTOP理由保持、FD/pipe終了、one-shot lock。
- 6受理前のobserver終了拒否、正常終端でidle worker退出前にobserverを意図的に終了・native waitする順序、異常exit拒否。

初回はNumPy.testingのhardware subprocess呼出をguardで拒否し12 error（spawn実行0）。
fixtureをprocess不要のallcloseへ置換し、35→41→44の限定回帰で補強した。旧失敗logはprivate evidenceに保存する。
実H4 compiler出力・native live-process timing/退出競合・性能/peak memoryは未測定。
local人工合格をproduction合格、CI、旧compiler bit同一性、科学的採択へ格上げしない。

## 入力・profile・資源・承認

.70/.80/1.40 ÅのNPZ計41160798 bytesは期待streaming SHAに一致。
generation-freeze 27115 bytes/SHAも一致し、JSON metadataだけからH/DF/state/input canonical identityを再照合した。
array payload byte SHA/dtype/shapeを6 taskへ転記したが、科学arrayのdecodeは0。
existing private environment/compilerは元固定profileと一致し、install/upgrade/共有設定変更0。
compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、num_processes1、approximation_degree1等の元explicit optionsを維持する。
旧hostとのoutput完全同一性は未検証のまま。

CPU候補worker0/driver1/observer2、1 worker、driver/worker各8GiB、observer AS256MiB/RSS64MiB、
admission120.25GiB、stage1h/task10min、output256MiB・fresh320MiB/256inodesは未承認proposal。
旧carry73023件/13615088338 bytes/268318.34715263546秒、旧actual上限74805・charge17GiBを保持する。
追加6件込み最大73029件。wallとRAMのcapは予測ではない。
snapshotの空き・CPU busy・quotaは権利/予約/launch成立ではなく、fresh gateで再確認する。

plan/authはapproved=false、runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。
retry=false、automatic_scope_expansion=false、next_stage_authorized=false、mandatory_stop=true。
実numeric FPは承認後のbuildまでnullで、technical bindingとruntime sealを区別する。

## GPTへ戻す論点

1. B0+Iを研究の別baselineとして採用するか（現時点では未採択）。
2. 代表6 full-wrapperの実測を行うか。source/入力/profile/予算/資源/CPUとone-shot STOPについて別の利用者承認・最終review・launchが必要。
3. 6件の実Δ_c/Δ_sをどの範囲の科学的解釈へ用いるか。397点全体・120補助構成のwinnerは6件では確定しない。
4. 10/240案、独立確認、希少次数診断、claim範囲を後から別判断するか。自動拡張しない。

H8の計算/worktree/source/grant、他job、共有環境、既存venvへの変更0。
SOURCEとsourceを変更しない準備bundle commitを分け、認証できればnon-force push・remote byte照合する。
認証不可なら設定を変えずlocal commitと手動pushを報告する。準備が完成しても実H4を起動せず、報告後mandatory STOP。
