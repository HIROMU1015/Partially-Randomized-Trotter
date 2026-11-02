# H4 certificate gate：技術段階Tのschema・記号的checker基盤

2026-10-23。利用者の「次の作業に進んで大丈夫」を、[前回最小契約案](../artifacts/non_df_h4_certificate_gate_audit/2026-10-23/minimal_next_contract_DRAFT.json)の次工程Tの承認として実施した。専用branchは `codex/non-df-h4-certificate-gate-foundation-20261023`、基点監査commitは `0369fa2a05aac7820e1c0f7aa7e55f9d048a0efa`。**技術SOURCEは `a66cefb497cb83c13158c9d4ab926202f791309f`**。C科学SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d` と旧科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12` は変更していない。

今回、証拠のscope・SHA・対象軸と、閉じた記号的規則を検査する基盤を追加した。保存値のラベル、ファイルの同一性、抽象的な式の成立を、実H4の科学certificate受理と分離する。**正式受理の全検証器が完成したという意味ではない。** 特に、実H4の数値包含とliteral/compiled programへ結ぶproof adapterは未実装であり、新しいSOURCEでの後続作業が必要である。

## 実施範囲

- schema、記号的phase/control/basis/compiler規則、人工・mockテストを新規実装した。
- 既存監査の40軸のmetadata・SHA・schemaだけを読み取り照合した。C resultの科学値はbytesをhashするだけで、再評価・再解析していない。
- 旧の科学run、SOURCE、raw、結果、grant、ledger、marker、STOP、compiler実装、shot会計実装を変更していない。
- 実H4数値計算、高精度・区間計算、sampling、RNG replay、compile、certificate生成・正式受理、N/G再会計は0。L/K/Aを実行していない。
- 主担当自身が実装と検査を行った。独立科学レビュー・最終evidence freezeではない。

対象の科学条件は既存のlinear H4/隣接1.00 Å/STO-3G、8 system qubit、Nα=Nβ=2、Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body、S2/T=.8/q8/r4/K4、4方式×5精度×cosine/sineのまま。他系列のH2 S2/Yoshida4研究を混合しない。

## 実装した境界

[library](../src/trotterlib/h4_certificate_gates.py)、[CLI](../scripts/verify_h4_certificate_gates.py)、[人工テスト](../tests/test_h4_certificate_gates.py)を追加した。科学依存をimportする `trotterlib.__init__` を経由せず、CLIは固定したlibraryファイルを直接読み込む。stdlibだけで動き、科学APIやshot会計への呼出経路を持たない。

| 対象 | 今回チェックすること | 科学的な意味の限界 |
|---|---|---|
| 40軸・14 gateのschema | case/精度/軸の全組合せ、一意性、gate集合、型、status語彙 | TARGET_MET/VERIFIEDという保存ラベル自体はproofではない |
| scope | binary readout、encoded event、literal wrapper、compiled program等の別 | readout支持集合をDF norm証明へ流用しない |
| 証拠 | 相対path、symlink拒否、SHA、固定SOURCE、target digestへの束縛 | hash一致は数式・誤差上界の証明ではない |
| 型付きlemma | 以下の閉じた規則と対象軸digestの照合 | abstract symbolと実H4行列/回路との対応は証明しない |
| 統計仮定 | PF/PR共通のfresh independent shots、PRのIID uniform-word | PCG64の無条件独立性証明を新必須条件に追加しない |
| compiled program | compiler対象が未定か明示し、program hashには実体の証拠を要求 | prewrapper fingerprintはpostcompiled programの証明にならない |
| 受理境界 | 実行・正式受理・N/Gのflagがfalseであること、gate別blocker | 現SOURCEではreal-H4 proof adapterがなく、受理は提供しない |

`validate_bundle` は宣言された `saved_claim`、検査済み証拠hash、`checked_lemma`、不足するproof接続を別fieldで返す。実H4 proof adapterの未実装を `H4_BINDING_AND_GATE_PROOF_ADAPTER_MISSING` 等で明示する。検査完了と、全gate証明済み/科学certificate受理を同じbooleanにまとめない。人工domainでも正式certificateは発行しない。

## 閉じた記号的規則

積は行列順序、右端が先に作用する。四分の一回転位相は整数mod4、一般位相はreal symbolの整数結合で保持する。実数値のtrig・行列・固有値計算は行わない。free wordの隣接inverse cancellationだけを使い、未実装の可換性・compiler passの成立を推測しない。

| 規則 | 検査する命題 | 残す前提・制限 |
|---|---|---|
| BINARY_READOUT | b∈{0,1}, Y=B(1−2b) の支持集合は±B | Bが有限非負。Cのencoded signalとの対応は別 |
| WORD_UNITARITY | exact Pauli/H、ideal RZ、unitary symbolの積とadjoint | RZ角はreal、一般symbolはexact unitaryという明示前提。丸めmatrix/P5/任意係数は拒否 |
| CONTROLLED_WORD | control=0はI、control=1はphase込みU | system global phaseを落としたbranchは拒否 |
| BASIS_CONTROL | B† diag(I,D) B の両branch対応 | Bのexact unitary仮定を残す。inverse順序を保持 |
| HADAMARD_WRAPPER | H-CU-HはRe、H-CU-Sdg-HはImのp0−p1 | normalized state、exact CU、ideal Born readoutの仮定 |
| WORD_EQUALITY | 支持するword reductionによるexact等価性 | 全Qiskit compilerの証明ではない。未対応rewriteは不受理 |

従って、Pauliのexact algebraとDFのbinary64行列を同一視しない。DF基底を `UNITARY_SYMBOL` として人工入力すると、その命題には `EXACT_UNITARY_SYMBOL:...` が残り、実H4とのbindingは `NOT_PROVED`。そのlemmaをH4受理へ転用してもblockerが残る。丸めDF行列をunitaryとして新しく仮定したcertificateは作っていない。

## 検証・保存40軸との接続

最終人工・metadata・mock CLIテストは **67 PASS、FAIL/ERROR/skip0**。そのうち1 testはX/Y/Zの長さ0〜4と4位相、計484個の人工wordをGaussian整数の2×2 oracleと照合する。H4ではなく、小さい人工例のexact演算である。wrong phase、inverse順序、sine符号、scope混同、別case/精度/軸へのlemma流用、偽SHA、source不一致、重複JSON/軸、symlink、非有限値、実行flag、旧namespaceへの出力を拒否する。

開発時の初回65件には1 ERRORがあった。seal拒否テストで、変更hashの照合前に別ファイルのGit確認へ到達していた。全コードhash照合をGit確認より先へ移し、拒否時にGitを呼ばないことを確認した。修正後65件PASS、追加oracle等を含む最終67件PASS。初回記録を削除せず `tests01.*`、修正後を `tests02.*`、最終を `tests03.*` に保持する。

技術SOURCE固定後、[SOURCE.json](../artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/SOURCE.json)で新規3 fileのcommit bytes/SHA、Python executable/version/SHA、前回監査inventory SHAをsealした。sealと矛盾するコード、別commitのbytes、runtime、監査inventoryは拒否する。この技術sealは科学実行grantではない。

封印済みCLIを1回実行し、[check01/preflight.json](../artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/check01/preflight.json)と[bundles.json](../artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/check01/bundles.json)に40軸×14 gateを保存した。**CLI exit0はmetadata検査の完了を意味し、科学的受理ではない。** 全40軸で正式受理false、旧N/G=null、数値目標TARGET_MET/64 bitを保持。旧VERIFIED fieldを全て新証明として受理したり、元の監査matrixを書き換えたりしていない。

出力は専用の `artifacts/non_df_h4_certificate_gate_foundation/` 配下の未使用directoryに限定する。既存outputは排他的に拒否する。次は実行済みコマンドの記録であり、同じoutputへの再実行指示ではない。

```bash
python3 -B scripts/verify_h4_certificate_gates.py \
  --seal artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/SOURCE.json \
  --output artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/check01
```

## 残存課題と担当

今回完成したのはscope-aware schema、閉じたlemma検査、証拠同一性の照合、受理を遮断する技術境界である。**real-H4 proof adapter、全compiler pass保証、結合上界の科学的検証、正式受理APIは完成していない。** これを解消するために全40軸を再計算する必要があるとは判断していない。

1. **ユーザー**：次に進める範囲を選び、compiler対象を旧programと新programのどちらにするか判断する。局所数値診断L・compiler検証K・正式受理A・N/Gは今回の承認に含めていない。
2. **Codex（承認後）**：証拠producer/verifierと実H4のmodel/gate/program bindingを別SOURCEで実装し、固定gateへ接続する。既存C/normalization保存証拠を再利用する。
3. **Codex（限定数値診断の具体的承認後）**：必要なら148種類の局所gateのforward/inverseと11基底に対象を絞る。精度、誤差の二重計上回避、call予算、入力hash、SOURCE/runtime、新grant、native witness、STOPを先に固定する。現時点で診断を開始しない。
4. **Codex（別承認後）**：実H4 proofが揃った場合に、固定IID条件付きで受理を検査する。その後のN/G会計も別承認・別STOP。

## 保全・終了

元H4 native終了未証明/監視原因UNKNOWNと、既存C run native PASSは独立して保持した。今回のmetadata CLI exit0を、そのいずれかのnative証拠へ代用しない。旧科学runのretry/resume、消費済みgrantの再使用、新しいreceiptによる過去の監視欠落の置換は0。

2,381既存ファイルのSHA保全と、新規SOURCE/結果の保存を終了監査へ記録する。[artifact入口](../artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/README.md)、[引継ぎ](../artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/CONTINUATION.md)、[公開記録](../artifacts/non_df_h4_certificate_gate_foundation/2026-10-23/publication_status.json)。更新共通AGENTSにより本branchのcommit/pushを行う。公開失敗時は原因と再実行コマンドを最終チャットへ示す。

**技術段階T完了後mandatory STOP。L/K/A・新しい実H4数値計算・正式certificate受理・N/Gへ自動移行しない。**
