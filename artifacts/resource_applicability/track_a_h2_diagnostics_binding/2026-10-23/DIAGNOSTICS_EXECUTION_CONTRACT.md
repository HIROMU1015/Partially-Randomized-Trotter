# DIAGNOSTICS：固定実行契約と受入証拠

状態は**技術封印準備済み・科学実行未承認/未実行**。機械可読の[実契約](DIAGNOSTICS_contract.json)と[preflight](preflight.json)を正とする。承認receipt・grantは存在せず、記載run_rootも未作成。これは科学実行を指示する文書ではない。

runner SOURCE `35444654f9159a5d85debdbc577feab0590e2610`、acquisition `4ec26921c6872ed6e8667f167074e1e405cf1a3f`、provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` を役割別に固定した。前段production SOURCE59e5dddd・基準d405b880は保全する。最終資料commitでrunner SOURCEを置き換えない。

## 固定4候補・条件

| 候補 | 方法 | q | prefix |
|---|---|---:|---:|
| B1_p3_q4_2nd | Full S2 | 4 | 3 |
| B1_p3_q2_4th | Full Yoshida4 | 2 | 3 |
| H2O32_c7ec590a32a859bc | PR S2 | 4 | 1 |
| H2O32_8a2ecdd0294ffb74 | PR Yoshida4 | 2 | 1 |

linear H2/隣接1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。PR K4/ABS_TIME_PER_OUTER32、通常signed bias、symmetric-directional、cos/sin full Hadamard wrapper。compiler条件は前段同様Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend・coupling・layout・routingなし。DIAGNOSTICSはwrapper compileを呼ばない。H/state/DF/PF係数/負時間/候補/q/r/K/費用指標は変更しない。

## 変更しないexact予算

| kind | call |
|---|---:|
| bootstrap | 1 |
| setup | 1 |
| prepare | 2 |
| diagnose | 4 |
| 内部path_signal | 8 |
| reference_rebind | 4 |
| native_validation / compose / lowering / wrapper | 各0 |
| compile / transpile / simulate | 各0 |
| request_factory / rte_event / occupation_trajectory | 各0 |
| allowance / accounting / ranking | 各0 |
| 新full-H reference / quantum shots / q探索 | 各0（実装分岐にも含まない） |

上位entryと内部callを併記し、表を独立実験数として合計しない。単位は前段[CALL_CONTRACT.md](../../track_a_h2_common_production/2026-10-23/CALL_CONTRACT.md)を維持する。bootstrapは科学import集合、setupは保存snapshot restore/H/state/basis/occupation構築、prepareはprefix3/1各1回、diagnoseは各候補の2 pathとreference再結合を含むjoin。Gaussian gate/actionの数はreceipt/workに記録し、追加prepare callとは数えない。

## 実行後に必須の証拠

| 証拠 | 保存先・解釈・受入条件 |
|---|---|
| 2共通prep | results/preparation_3.json、preparation_1.json。prefix、prep hash、rotation byte hash、gate/hex angle/wires、operations hash、native policy |
| 4候補×2 path | 各candidate_diagnostics.jsonとcandidate_vectors.npz。primary spectral、independent matrix expm/Taylor。共有setup/入力があり全誤差独立の証明ではない |
| raw/corrected | PRは列0 corrected、列1 attenuated raw。各pathのexp(log_B)でrawをcorrectedに戻す。Fullはcorrected1列/B=1 |
| normalization | log_B、2 path差、signed schedule。Y4の負時間とr割当を保存 |
| work/norm/trace | primary/independentそれぞれdeterministic/tail workと全stage trace。raw normはcorrected単位。Taylor term/sum・spectral basis内を含む |
| strict gate | corrected-unitのpath signal/vector差、log_B差、corrected-signal closureがすべてstrict `<1e-9`。phase alignment/postcompile correctionなし |
| reference | 各候補のcorrected−旧FP scalar、reference ID/error/binding、reference_recomputed=false、certified=false |
| scalar | Full constant、PR constant＋tail identity。scalar_applications=1、各path最後に一回 |
| native identity | candidate/prefixに対応するprep/operations hash、policy、SOURCE、入力digest。native/compiled correctnessはこの分岐では未評価 |
| baseline | 新signal/trace/vectorと保存baselineのlocator/SHAを別々に残す。旧Fullはsealed old_reference excerpt /old_signals/0, /old_signals/1、PRは受領signal_records_2.jsonの候補ID。今回baseline差の数値算術は0。後続で差を評価するなら別途承認範囲へ明記し、差を新uや順位へ自動変換しない |
| 完了bundle | completion.jsonにresult全file SHA・exact used・SOURCE/runtime/binding/policy・CONSUMED marker SHA。ledger各予約とcontract digestが一致 |

NPZ保存は同じ診断で既に計算したvector/basis/qindicesのfile出力であり、新しいpropagationではない。後続Full native correctnessに独立vectorを再利用するために保存する。今回はこのNPZを一つも生成していない。

## 隔離launch・承認・one-shot

preflightの実コマンドは[preflight.json](preflight.json)に保存。選択Pythonを `-I -S -B -X pycache_prefix=<未存在専用path>` で起動し、site/.pth/PYTHONPATH・未封印pycを使わない。科学実行時も同じlauncherに外部approval/grantを渡す。実行コマンドの自動提示・実行やgrant発行は今回行わない。

承認はissuer=USER_EXPLICIT、execution_authorized=true、**実契約canonical JSON digest**への結合、3政策の明示採択を必要とする。新grantは契約digest・approval digest・exact budget・専用run_root・新IDに結合しstate=UNUSED/retry=false。未使用rootのexclusive作成後、CONSUMED markerを科学import前に保存する。grantが保存ファイル上UNUSEDと書かれていても、消費済みroot/markerがあれば再使用不可。

ledger予約は失敗しても返金しない。未封印import、SOURCE/input/runtime/marker差、count超過、数値gate失敗、例外・中断はSTOP_PARTIAL_NOT_PASS。partial NPZ/JSONを完成扱いせず、retry/resumeしない。completionがあってもscience_GO=false、mandatory STOP。正常終了した診断も比較の自動承認にはならない。

今回のpreflightでは科学importの観測・検証をしていない。静的許可集合から外れたmoduleを通常import/直接source loader前に拒否し、backend import後のsetup前と完了時にloaded module origin/executable ELF mappingを検査する。未知依存が実際に要求された場合もbootstrapを返金せずSTOPし、準備の再封印と新承認の対象にする。

## 必要な明示承認

ユーザーが[POLICY_DECISIONS.md](POLICY_DECISIONS.md)の3政策・計数/経験的scope・fresh Python3.11 runtime・旧FP referenceの条件付き解釈を採択し、契約digest、4候補、上記exact budgetだけのDIAGNOSTICSを承認すること。新grant/rootとno-retry/mandatory STOPを含める。PR C witness、compile、sampling、会計、COMPARE、PR_TWO_NATIVE_CONTROLS、科学レビューの承認は含めない。
