# PR-Pauli LD2 B1-W：実回路配線・X/Y 2 compile

2026-11-15 JST、Codex D2。**承認済みの1 job・normal index0の1 path・X/Yの新規2 compileを完了した。観測pathのcorrectness gateはPASS、FAILなし。** X/YともRZ **2236**、CX **960**、total depth **1416**。公開SOURCEを束縛したC guardも両軸を受理した。取得・公開・束縛後、mandatory STOP。

今回の2 eventはどちらもorder0だった。**LD2実pathのorder2 productと非自明なevent位相は未観測（UNKNOWN）**であり、その実物理検証まで完了したとは認定しない。既存toy回帰のorder2検査は実LD2の観測と区別する。追加draw・別seed・追加path・retryは0。

## 固定SOURCE・条件

| 項目 | 固定値 |
|---|---|
| branch | `codex/d-pauli-b1-caller-prep-20261114`を継続 |
| 実行SOURCE | `1a7eed91dd96992cc6abcbfa1cd473a2044db16c` |
| 実装SOURCE | `bd1b1b2862da6d174fa521e7612051810a636e7f` |
| raw公開commit | `6c4709f85d0f28ce7c6957e9a70c85e378d8cad0` |
| 系 | linear H4、1.00 Å、STO-3G、保存tol-only DF10、8 system qubit |
| 方式 | DF head＋Pauli tail、LD2、S2、canonical m0、T=0.1、qPF=1、R=2、K=2、P=0 |
| 元configuration | `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3` |
| physical ID | `RMSE:babca160daaf91bf567324e3e3ce75d7f2b6c6b9658634d5042e970d4ddc444d` |
| H hash | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| state hash | `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b` |
| snapshot hash | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| LD2辞書 | 保存184項、SHA-256 `dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7` |
| 辞書・P0 | `4220f3006d0d8d413e195ed948e250578513b557` |
| A signal / B1設計 | `714247619afb4714fd663eb15c0470ef32bdc7af` / `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` |
| C planner・guard | `c628e9e8447c11c984064b922cbce6c52dc72d48` |
| cohort / index / seed | `D_PAULI_LD2_B1-W_M0_PROPOSED_V1` / 0 / `3788801539369037729` |
| route / lowering | `PR_PAULI_TAIL` / `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF` |
| compiler | Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17 |
| compiler profile | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |

既存caller `scripts/run_pauli_b1_prices.py --execute-approved-job JOB.json`を一度だけ使用し、新しい回路生成器・実行管理機構は作っていない。既存branch上の承認済み次段階として新namespaceに保存し、旧SOURCE・結果・STOPを保持した。共通運用指示2026-11-14 §8、§10.3–10.6を適用した。

[JOB](JOB.json)は元`JOB_TEMPLATES.json`のLD2 B1-Wを使用した。採択済みWに限って`adopted=true`とし、出力先と実行直前照合の状態を固定した。cohort名の`PROPOSED_V1`とdesign_sourceの`PROPOSAL_NOT_ADOPTED`は元提案の識別子として保持し、V/CALの採択を意味しない。schemaの`stage=CALIBRATION`も既存Wの表現であり、purposeは`WIRING_OR_VARIABILITY_ONLY`、実出力`C_INPUTS=[]`である。

## 実行直前の重複照合・再利用

[PREFLIGHT](PREFLIGHT.json)に対象資料・一致項目・SOURCE・remote refsを保存した。D2の現行SOURCE、B0の元trial/receipt、A設計と最新branch `1adb7e65a`、Cの価格計画・coverage、最新候補対応表 `0b4b2d941`を照合した。C対応表の当該physical ID・m0・X/Yはいずれも`PRICE_EVIDENCE_MISSING`だった。公開Pauli branchに別の取得成果はなく、同一候補の実行中local jobは見つからなかった。調査はこれら関連する公開記録と参照可能な同hostの記録に限定した。

| 保存資産 | 今回の採否 |
|---|---|
| 保存H/state/snapshot、LD2辞書・P0演算子 | hash/byte・prefixを照合して再利用。辞書の再収集・cutoff変更なし |
| B0 `00f74f86d`のnormal path0と価格 | LD1の別辞書・splitなのでLD2へ流用しない |
| 旧P3、Pauli-JW、DF-tail価格 | Aの既存照合を継承。law/K/head/辞書等が異なり直接価格として不採用 |
| native、wrapper、D2、compiler、sampler | 元処理を再利用。SOURCE byteとproviderの既存検査を保持 |
| 元normalization | 保存m0のB=1.0212069010293483、log B=0.02098516412709399を記録から読取り。再算出せず価格にも乗じない |

native/wrapperは`af284d85`、D2は`9c4006bf`の固定blobと一致し、C guardも固定版と一致した。元B0との実装差は既存B1のprefix・step seed・event metadata対応であり、今回code変更は0。metadata seedは元`H46_LONG_TIME_20261025`規則のcanonical ID/cohort/index0/outer0/piece0と一致した。新output/private/cacheは未使用だったため、必要な新規取得はLD2の1 path・両軸2 compileと確定した。

## 実pathとcorrectness

DF headはone-body＋保存fragment 0,1を元順序で保持し、保存head/tail block列との差はともに0。Pauli辞書の係数・順序・label/bit順を今回の回路のcorrectness gateとして検査した。tailと保存s_R付きPauli演算子の差は1.5077696299434903e-15。独立した保存P0再検査を別途起動していない。

保存s_R=0.2223759094107018はnative tail identityと一致し、c=2.29310124732とのtotal scalarは2.515477156730702。各native stepのscalar=0、総Tで最後のancilla phaseを一回だけ適用する既存gateを通過した。

| event順 | order | component / Qiskit label | 係数符号 | phase |
|---|---:|---|---:|---|
| 0 | 0 | 34 / `XZZXYZZY` | +1 | 1+0i |
| 1 | 0 | 40 / `IIXZIZXI` | −1 | 1+0i |

outer=0、signed_time=0.1、seedは上表の固定値。両eventのrotation_angle=0.10239561857008683、event_normalization=1.0105478222376951は保存値と一致した。D2 metadataのproduct tokenはなく、aggregate source phaseは0。詳細なapplication sequence、係数、label、D2 instruction mappingとphase ownerは[RESULT](run01/RESULT.json)に保存した。order2を強制するsamplingはしていない。

| gate | 誤差 | 許容値 | 判定 |
|---|---:|---:|---|
| low-control identity | 2.4128158003448048e-15 | 1e-9 | PASS |
| high-control leakage | 0 | 1e-9 | PASS |
| independent full-space vector | 5.290882962960572e-15 | 1e-9 | PASS |
| X Hadamard | 3.6637359812630166e-15 | 1e-9 | PASS |
| Y Hadamard | 7.771561172376096e-16 | 1e-9 | PASS |
| compiled vector X | 2.81820739483553e-13 | 1e-7 | PASS |
| compiled vector Y | 2.543103741646044e-13 | 1e-7 | PASS |

full-space oracleは全2^8次元の保存状態への作用照合であり、全入力operator同値性の認定ではない。LD2のorder2 product・非自明event位相はUNKNOWN。安全な既存回帰は[B1 metadata](TEST_B1_METADATA.txt)11件、[B1 toy](TEST_B1_TOY.txt)3件、[B0 metadata-only](TEST_B0_SAFE.txt)親8件PASS（その内部のtoy5件を含む）。toyに含まれるorder2/product/phase検査と実H4の証拠を混同しない。作者自身のlocal検証であり、別agentの独立レビューではない。

前回の範囲逸脱とそのログは元`1a7eed91`の`docs/research/d_pauli_b1_caller_preparation_20261114/SCOPE_DEVIATION.md`等を無変更で保持し、今回の科学証拠には採用していない。unfiltered回帰・`--include-saved-numerical`は実行していない。

## compiled価格・資源・公開

| axis | RZ | RZ depth | CX | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|
| X/cosine | 2236 | 596 | 960 | 516 | 1416 | 4539 |
| Y/sine | 2236 | 596 | 960 | 516 | 1416 | 4539 |

測定付きfull wrapper、9 qubit、state preparation費用は除外。X/Yを各一回compileし、cache hitは0。軸別fingerprintは異なる。

- X：`3cce8471c2094c66429b8570651920af3b585af620281ccaad59291f7fc45b8d`
- Y：`55e78572fd968efa8e3da2dbf9eeedf3902a63734ca4769a301346f912e6dee3`

[COMPILED_X](run01/COMPILED_X.json)、[COMPILED_Y](run01/COMPILED_Y.json)、[PRICES](run01/PRICES.json)を公開し、元wrapper/compiled QPYは専用private領域に保持した。所在・bytes・SHAは[RECEIPT_TRACE](run01/RECEIPT_TRACE.json)。全command wall **11.29秒**、peak RSS **631036 KiB**、swap 0、exit 0（[資源log](RESOURCE_TIME.txt)）。CPU affinity64、開始前MemAvailable約488 GiBを観測し、競合する同候補jobはなかった。新しい資源上限・共有設定変更はない。m0一回の性能を長roundへ外挿しない。

rawをcommit/pushした後、その公開SHAを既存`check_round_records`へ渡した。[SIDECARS_UNBOUND](run01/SIDECARS_UNBOUND.json)は無変更で保全し、別の[SIDECARS_PUBLISHED](run01/SIDECARS_PUBLISHED.json)を保存した。[C_GUARD_CHECK](run01/C_GUARD_CHECK.json)は両軸`IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`。physical ID・辞書/law・lowering・scalar・compiler・axis・証拠SOURCEのmetadata整合を確認したもので、比較policyや価格modelの科学的資格を認定していない。将来用のW receipt読取も[WIRING_METADATA_CHECK](run01/WIRING_METADATA_CHECK.json)でPASSした。

rawのremote SHAは`6c4709f85d0f28ce7c6957e9a70c85e378d8cad0`、treeは[PUBLICATION_CHECKS](PUBLICATION_CHECKS.json)に記載。後続の報告・束縛sidecarも同branchへcommit/pushし、最終remote SHA/treeは最終回答で照合結果を報告する。

## 科学的資格・ロードマップへの影響（§10.6）

価格は`DIRECT_COMPILED_FINITE_SAMPLE` / `FINITE_COHORT_ONLY` / n_paths=1、law_expectation=false。同一pathのX/Yであり、独立2標本ではない。schemaのmean欄はこの有限cohort一個の価格であって、Pauli law全体のE[C]や統計区間ではない。`C_INPUTS=[]`を確認し、較正には混入していない。

LD2の初回実配線・観測path作用・実compiled価格と、CへのSOURCE付き接続が完了した。ロードマップで更新すべき点は「B1-W未実施」からこの限定した完了への変更である。order2/非自明位相の実LD2観測、path分散・rare寄与・sampler law差・model/外挿精度・全round E[C]・MSE/N/Gは未資格のまま残る。ロードマップ本体は編集していない（このSOURCEとmainには同名ファイルなし）。

将来Vを採択する場合は今回の元cohort/index0/seed/公開SOURCEを保ち、今回を新しい独立標本として数えない。CALには別途の採択・実行許可と物理contextの維持が必要であり、今回のW取得はその実行許可を意味しない。[HANDOFF_C](HANDOFF_C.md)にreceiptを渡す。**B1-V、B1-CAL、fit、E[C]、MSE・N・Gへ進まずmandatory STOP。**
