# Track A AX-2B：H4限定科学検証manifestの固定・seal v1

2026-10-10 JST。利用者の「作業を進めて」を、直前に提示した**H4限定science manifestの固定・sealだけ**へ適用する。
[GPT独立レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) §21、[既存結果前契約](track_a_ax2b_h4_prelaunch_contract_v3.md)と[保存receipt再監査](track_a_ax2b_h4_native_receipt_reaudit_v2.md)を入力とする。
新しい科学実行・H4-P再実行・native準備・NPZ数値配列decode・sampling・回路構築・compileを行わない。
sealは実行条件の固定であり、認可ではない。`science_authorized=false`、`launch_allowed=false`、`H4_LIMITED_NOT_AUTHORIZED`を維持する。

## 来歴と固定対象

[metadata専用seal入口](../../scripts/resource_applicability/seal_track_a_ax2b_h4_limited_v1.py)を追加し、凍結された科学launcher・backend・過去のfreezes/STOP/科学結果を編集しない。
新入口は`--execute`/`--worker`/`--authorization`を持たず、approved grantも生成しない。
ライブラリclosure全件のlocal/Git bytes、新seal工具・testsのGit bytesを別々に固定する。
数値ライブラリimportを禁止し、保存JSON・Git blob・file hash・NPY header/Unicodeだけを扱う。

| 記録 | 正本と役割 |
|---|---|
| H4-P実行source | `7877131e764f5ce8b296cbbfa9ff3e859d04a8f2`。旧実行closure170件。新科学sourceへ読み替えない |
| H4-P結果保存 | `1173dd3342e88239458bcce17ae6a4047f8f1fef`。原16 fileのbytesを再照合 |
| 保存再監査source | `941eda0b22bfedda10ece5ed4cdefad77cf3d2b2`。audit closure172件、継承freeze187件 |
| 再監査保存 | `ab98bed720bc9d3ba5d45c619f1a1575fccef4a1`。[保存receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10/reaudit_receipt_v2.json)のlocal/Git bytesと新metadata再確認のsemantic digestが一致することを要求 |
| 科学source/manifest | 後掲の固定記録。現在のlibrary＋凍結v2科学runnerのclosureを別に固定。seal入口/testsは別工具hashも保存 |
| 新準備inventory | [seal inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/seal_inventory_v1.json)。既存scientific manifestのstatusや結果を更新しない |

元親の`H4_NATIVE_RECEIPT_STOP` / `TERMINAL_INVALID:JSON_INPUT_SIZE`は不変。
新再監査PASSを元launch成功へ変更しない。保存readerは元16MiB aggregate budgetで2 pass照合する。
新manifestへ取り込むのは、再監査済み`original_coverage_binding`のexact JSON値であり、旧実測compiled costから推定しない。

## 固定する科学scope

linear H4 1.00 Å、STO-3G、legacy DF rank12・generation-prefix、8 system qubits、Nα=Nβ=2・sector36、T=0.8。
保存binary64 DF係数の数学的Hamiltonianと、指定保存vectorの数学的正規化がtarget。
真の基底状態、元積分Hamiltonian、別geometry/rank/時間・新状態へ置き換えない。
[既存snapshot](../../artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz)は8,652 bytes、SHA-256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
file bytesと8 NPY memberのheader/Unicode metadataを照合し、数値payloadはdecodeしない。

| 登録cell | method/PF | L_D | q | R/r/K | ordinary構造instruction上界 | directional構造instruction上界 |
|---|---|---:|---:|---|---:|---:|
| H4_B1_S2_q1 | B1/S2 | 12 | 1 | — | 2563 | 4102 |
| H4_B1_S2_q4 | B1/S2 | 12 | 4 | — | 10237 | 16393 |
| H4_B0_q4 | B0/S2 | 6 | 4 | — | 5389 | 8497 |
| H4_B2_K2 | B2/S2 | 6 | 4 | 8/2/2 | 7161 | 10269 |
| H4_B2_K4 | B2/S2 | 6 | 4 | 8/2/4 | 8345 | 11453 |
| H4_B3_K6 | B3/S2 | 0 | 4 | 8/2/6 | 4665 | 4725 |
| H4_B1_S4_q1 | B1/global S4 | 12 | 1 | — | 7679 | 12296 |
| H4_B1_S4_q4 | B1/global S4 | 12 | 4 | — | 30701 | 49169 |

δ=T/q=0.8/0.2。random microtime0.1、代表event half/undo±0.05、global S4の負時間を含む登録集合を維持。
表は保存native operation/spec metadataから凍結構造式で得た上界。実compile、expanded gate数、物理資源やcompiler RAMの上界ではない。
179 primitive-ID/time組×保存state・first/last sector columnの3 probe=537予定作用をexact boundsへ固定する。
今回のprobe作用は0。nested schedule内の旧`molecular_probe_plan_sealed=false`は原metadataのまま保持し、futureの実科学probe実施済みとは扱わない。

H4-N/A：既存sourceによるsector binary64 expmと独立occupation全36列照合、MP80/120桁各8 cell、stage state/norm・reference差・raw/corrected/logB・signed誤差分解。
独立性はoccupation Hamiltonian構成と別精度・Taylor前進和にあり、共通PF iteratorやprepared orbital decomposition全体の独立証明は主張しない。
H4-E：B2 K2/B3 K6各order0/2の4代表event群、100 control probe予定。generation順first nonidentity、両control/branch/axis・algebraic event検査を既存sourceのまま固定する。
代表eventの明示回路構築・statevector検査は**将来の別科学認可scope**。sampling/compileや全分子event平均列挙は含めない。
H4-M：経験的差を総u認定やshot確定へ昇格させない。N/G=null、accuracy UNDETERMINED、総allowance未認定。

## 固定する古典計算条件

| 項目 | 既存planから固定する値 |
|---|---|
| CPU/worker/BLAS | CPU3 / worker1 / BLAS1。計画上の指定でありOS予約・科学worker起動ではない。launch時availability再照合必須 |
| 環境 | Python3.11.0rc1、numpy1.26.4、scipy1.14.1、mpmath1.3.0、qiskit1.3.0、openfermion1.6.1。dist metadataで照合しimportしない |
| phase/total wall | input_reference900秒、correctness1800秒、explicit estimator（旧phase名wrapper_cost）300秒 / total3000秒 |
| address space | 8GiB。RSS実測や予約量ではない |
| output/log/diagnostics | 512MiB / 64KiB / 1024。既存writerはterminal領域を予約 |
| reference/deterministic/tail | reference matvec合計10000・per-action20000、deterministic100000/cell、tail896/cell |
| primitive/control/stage | primitive2000、control200、stage4096/関数呼出し（全path合計） |
| instructions | untranspiled1,000,000 |
| sampling/compile/solver/quantum shots | 0。明示代表event buildだけは将来のH4-E科学scope |
| 専用future output | `artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1`。今回directoryも作らない |

これらは評価のための古典計算上限であり、量子回路資源の測定結果ではない。
source/input/env/coverage・bounds不一致、caps超過、nonfinite/Hermiticity/sector/phase/register不整合で既存STOP条件を適用する。
retry/resumeなし、cap自動増量や結果依存間引きなし、全terminal後mandatory STOP。保存元結果の再fit/再最適化なし。
検証関数の閾値・MP桁数・stage規約は科学source hashに固定し、新しい閾値や科学比較条件を追加しない。

## sealと認可の境界

[新manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/sealed_preparation_manifest_v1.json)は凍結v2 schemaを保ち、`execution_plan_sealed=true` / `coverage_binding.sealed=true`とする。
`science_authorized=false` / `launch_allowed=false` / `H4_LIMITED_NOT_AUTHORIZED`は維持。
[seal監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/seal_audit_v1.json)にsource/input/environment/boundsとmanifest digest/file SHAを記録する。

専用outputの相対・絶対pathはmanifestの`intended_exclusive_output`へ固定する。
**凍結v2 launcherはこの追加fieldを直接enforceせず、別grantの`exclusive_output`をenforceする。**
後続の認可作成時にはgrant outputをこの固定pathへ一致させ、manifest digest・authorization file SHA・CPU3・retry/resume=falseを結び付ける必要がある。
今回grantはnullであり、実行コマンド・approved grant・worker claimを作らない。このstageのsealを実行認可や起動成功と呼ばない。
後続launchでsource file集合が変わった場合も凍結gateが拒否する。無関係な追加sourceを混ぜて自動rebindingしない。
future outputの親directoryは未作成でよく、実行時の限定準備で親だけを作成しても`launch_v1`はexclusiveとして保護する。

## 合成検査と保全

[専用tests](../../tests/tracks/resource_applicability/test_ax2b_h4_limited_seal_v1.py)35件がlocal pass。
合成JSONのSTOP/coverage/instruction/CPU/env誤り、copy保全、published bytes不一致、未認可gateのI/O前拒否、CLI実行option拒否を検査。
[JUnit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/synthetic_tests_v1.junit.xml)はlocal工程証拠であり、科学検証・immutable CI・精度認定ではない。
旧テスト/科学検証を再実行しない。[保全監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/preservation_audit_v1.json)でpreexisting files/dirty/未追跡/Track B・root reviewと旧STOP/freezesを確認する。
今回source工具/tests・準備metadata・文書・索引だけを既存branchへcommit/pushする。旧科学manifestや状態を変更しない。

現在のSTOP：`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`。
次の科学実行はこの固定manifestを対象とする別の明示認可後に一回だけ。H6入力生成・pilot、本検証、H8/GPUや科学GO/STOPは今回開始・代行しない。

## 固定完了の記録

science/工具source commit：`61091c2cb00eb871d7a692b125219d34d99cc923`。future science closure170件とmetadata工具2件、継承187件＋新2件の準備freeze189件をlocal/Git bytesまで確認した。
旧native execution closureも170件だが、現在のclosureとはfile集合が異なる。同じ件数を同じ実行sourceの意味にしない。
manifest file SHA-256：`679511b859630992870be50e2ec4d15fa24c4794ddec2bc8b6d127f3e65a58c2`、semantic digest：`4dc632f579e69b9aa98a97522169846a1653ffbb4680036c76823d6993d3907d`。
seal statusは`H4_LIMITED_PLAN_SEALED_NOT_AUTHORIZED`。未認可gateは`SEPARATE_EXPLICIT_USER_GRANT_REQUIRED`でI/O前拒否。専用science output・approved grant・worker claimは作成していない。
[source対応](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/source_binding_audit_v1.json)、[準備freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/preparation_source_freeze_v1.json)、[保全監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/preservation_audit_v1.json)へ照合結果を保存する。
旧2,656 fileとroot側5 review文書を保全。新科学計算・H4-P再実行0、既存dirty/未追跡はcommit対象外。mandatory STOPで別の科学実行指示へ戻す。
