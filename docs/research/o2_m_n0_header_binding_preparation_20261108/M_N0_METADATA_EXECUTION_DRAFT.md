# 案M：N1 n=0一回metadata-only科学実行契約草案

**未採択・未実行。今回の許可は技術準備だけである。** 次回の明示的な実行命令があった場合にのみ以下のpassを起動する。新しい承認証明/grant発行・manager基盤は要求しない。

## 対象と固定SOURCE

N1結果`744e1a09661ebba6841fe5faf456f88c41c47780`、N1科学SOURCE`b6dbec9f80da35a88fba28795ad02ca74c9adfa0`。保存linear H4／1.00 Å／STO-3G／legacy DF12、8 system qubits、L_D=3、m0/q1、δ=0.02、r1/K2、coefficient_atol=0、N1 actual finite-grid Q・selected full policyを維持する。全4電子sectorは既存N2の指定で、このpassはstateを読まない。

対象は**n=0 eventの唯一のrotation application slot**。stable IDは`m0/q1/r1/order0/application0/rotation`。他order/slot/round/basisの再選択や探索を行わない。n=0の条件付き部分への適合確認であり、元targetからn=2等の寄与を捨てる契約ではない。

本草案と同じ公開commitのbinder/実行入口を実行SOURCEとして固定する。最終SHAは今回の最終チャットおよび公開branchから取得し、実行前にその完全SHAを採択する。自己SHAを本文に埋め込む追加commitはしない。実行HEADと対象SOURCEファイルのbyte、Python3.12.3/NumPy1.26.4/Qiskit1.3.0をpreflightする。

## 入力preflight

読込先は`artifacts/o2_a1_new_target/2026-11-06/n1/`の次の5ファイルだけ。以下のGit blobはfixed treeのobject名を照合して固定した。実値を使う群計算は今回実行していない。

| file | 固定Git blob |
|---|---|
| SOURCE.json | `9cf56d6570133bde49a692ca1cb7cb064382a0a1` |
| CONFIG.json | `ad272e31d0dfc2b5087721a2ed7ab488d663724e` |
| LAW.json | `15b40241ba648c0f8750e087425d3baeb053730f` |
| POLICY.json | `c089717e43b3ab5b1a637e802471a2a6bb40ba9c` |
| PREPARATION.json | `d7cf105c31b30100ee16dc4fefe41999cd5c009f` |

実行時は各byteからGit blobを照合し、その後だけJSONを解析する。SOURCEのscope、tail/preparation/policy/law、ruleとselected定義の内容fingerprint、basis metadata hash、system幅・順序を確認する。実入力のidentity項混入は元のextracted-identity契約との不一致としてSTOP。N2・ground・state・spectral・旧engine認証は読まない。

## 一回のcall境界と入口

[m_n0_metadata_execution.run_metadata_pass(root, fresh_output, expected_source_commit=...)](../../../src/trottertracks/algorithm_codesign/m_n0_metadata_execution.py)を**一回、retry 0**で実行する提案。薄いCLIは[run_m_n0_metadata.py](../../../scripts/tracks/algorithm_codesign/run_m_n0_metadata.py)。別途実行承認後の形は次の通りで、今回このcommandを実行していない。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python \
  scripts/tracks/algorithm_codesign/run_m_n0_metadata.py \
  --source-commit <今回公開する完全SOURCE_SHA> \
  --output artifacts/o2_m_n0_header/2026-11-08/metadata_run01
```

同じ固定SOURCEのcleanな独立worktree、fresh outputを使い、既存directoryの再使用・上書きをしない。rootはrunnerのcheckoutから決まり、保存N1 directory配下への出力を拒否する。一回制約はユーザーの科学契約と保存STOPで維持し、別outputでの再実行を自動許可する意味ではない。

passは5ファイルを一度読んで照合し、`bind_n0_headers`一回、`summarize_binding`によるpartition一回だけ。有限catalog内の全labelを一度記述化する。selected定義/basisとD blockはそのために必要なmetadataだけ扱い、全path/event組合せや全system outcomeを列挙しない。実行する算術には、保存対角値から既存primitive descriptorを得ること、保存tauの元SOURCE規約でのphi生成、actual整数質量のFraction化/群和/qが含まれる。単なるbyte閲覧ではなく、この範囲の新しい科学的metadata集計として許可を受ける。

## 保存内容と判定

| 出力 | 内容 |
|---|---|
| SOURCE.json / INPUTS.json | 実行SOURCE・環境・SOURCE hash、固定N1参照、input blob/SHA256 |
| CALL_COUNTS.json / CALL_START.json | 初期countsとbinding開始。開始前に保存 |
| HEADER_BINDINGS.json | input identities、全labelの適格/zero/失敗情報、A/B/V literal、φ/sign/phase/wire/order/target対応 |
| GROUPS.json | 共通署名、label集合、p_g、Q(n=0)p_g、exact conditional q、signed function、明示adjoint情報 |
| RESULT.json | 下記判定、UNKNOWN。binding失敗なら取得済みlabel記録・未処理UNKNOWN |
| CALL_COUNTS_FINAL.json / STOP.json | 実際のattempt数と禁止call=0、retry0、mandatory STOP、下流未開始 |

非自明群は「正labelが2つ以上、共通factorizationのliteral十分条件を満たし、signed diagonal functionが2種類以上」と定義する。別名labelだけでは非自明としない。

| 判定 | 意味と次の処理 |
|---|---|
| NONTRIVIAL_GROUP_FOUND | このslotで非自明群を取得。分散/価格/科学的GOを保証せずSTOP |
| NO_NONTRIVIAL_GROUP_IN_SELECTED_SLOT | この固定slot/記述では取得できず。一般MのNO-GOや別slot探索へ移らずSTOP |
| ONLY_KNOWN_DIRECT_MEASUREMENT_STRUCTURE | 群ごとの別欄。A=B†がliteralで明示でき、既知ancilla-free測定との比較を優先課題として残す。非自明群の有無と独立 |
| BINDING_INCOMPLETE | 必要metadata/operation/phase対応が得られない、未対応gate、予期しない例外。推測/近似で補わずSTOP |
| INPUT_OR_POLICY_MISMATCH | input/SOURCE/環境、scope/law/policy/selected identity等の不一致。元SOURCEを修正して救済せずSTOP |

正supportを消す・群ごとに都合のよい再正規化をする処理はない。正label全体の群質量和=1を確認し、zero-grid labelは元lawと記録に残す。A=B†をliteralで認識できなければUNKNOWNであり、適用不能の証明にはしない。

## 禁止・STOP

N1/N2再生成、Hamiltonian/DF/CDF再計算、state/ground/eigh、実H4回路build/unitary matrix/state evolution/Born、RNG/shot、compile/native synthesis、全zのr/m、分散/N/費用、価格による群選択、別slot/round探索を行わない。群が見つかっても量子pilotやengine誤差認証へ自動移行しない。

出力衝突は処理前拒否。preflight/入力/対応不整合で一回attemptを中止して保存STOP。取得できなかった項目はUNKNOWNのまま。SIGKILL/電源断等でもfinalizerが必ず完走する保証はなく、開始記録と出力衝突により保全し、勝手にretryしない。新しい持続managerや承認証明を作る必要はない。

次に必要な明示承認は、**この公開SOURCE・固定N1・一つのn=0 slot・一回metadata-only pass・retry0・上記出力/STOP**の実行許可である。今回は未許可なので、実N1を投入せずSOURCE公開照合後STOPする。
