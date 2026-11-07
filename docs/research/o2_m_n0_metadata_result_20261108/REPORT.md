# O2案M：N1 n=0 metadata-only一回実行の結果

2026-11-08 JST／SERVER1 GPUA100-gleap-01。**一回のmetadata-only passは正常完了し、判定は`NO_NONTRIVIAL_GROUP_IN_SELECTED_SLOT`。この固定slot・保存selected policy・literal十分条件では、複数正labelをまとめる非自明群を取得できなかった。** 科学retry・別slot探索・量子測定へ進まずmandatory STOPした。

## 固定条件・一回の実行

科学実行SOURCEは[`39d839063c53d949342af4a0541b4865bdd054ed`][SOURCE]。ユーザーの明示許可により、同SOURCEの[実行草案][CONTRACT]の範囲を一回だけ実行した。専用branch `codex/o2-m-n0-metadata-run01-20261108`、独立worktree `/tmp/o2-m-n0-metadata-run01-20261108`は、入力前にSOURCEと同じHEADでcleanだった。SOURCE・binder・serializerは修正していない。

入力はN1結果`744e1a09661ebba6841fe5faf456f88c41c47780`、N1科学SOURCE`b6dbec9f80da35a88fba28795ad02ca74c9adfa0`の保存linear H4／1.00 Å／STO-3G／legacy DF12、8 system qubits、L_D=3、m0/q1、δ=0.02、r1/K2、coefficient_atol=0、actual finite-grid Q、selected full policy。対象slotは**`m0/q1/r1/order0/application0/rotation`**だけ。全4電子sectorは既存stateの指定であり、このpassはN2/stateを入力にしない。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python \
  scripts/tracks/algorithm_codesign/run_m_n0_metadata.py \
  --source-commit 39d839063c53d949342af4a0541b4865bdd054ed \
  --output artifacts/o2_m_n0_header/2026-11-08/metadata_run01
```

CLI exit codeは0、stdoutは判定文字列。既存docstringのSyntaxWarningがstderrへ出たが、SOURCE修正や再実行はしていない。実行環境は保存SOURCEのPython3.12.3、NumPy1.26.4、Qiskit1.3.0と一致した。新しい人工testは不要のため実行していない。

## 取得した結果

| 項目 | 結果 |
|---|---|
| 個別header | 正grid質量232 labelがELIGIBLE_HEADER、zero-grid 92 labelを別記録で保全 |
| 群分割 | 232群、すべて1 label。非自明群0 |
| 群内actual q | 各単独群は(1)。群質量は各labelの元actual質量のまま保存 |
| 外側n=0質量 | `9006586475752217/9007199254740992` |
| full context質量 | Q(n=0)×群質量を各群に保存。元lawの他orderを削除しない |
| A=B†の明示構造 | 全群`UNKNOWN_NO_LITERAL_ADJOINT_MATCH`。非適用の証明ではない |
| source/input preflight | 各1回、固定SOURCE/環境とN1入力5 blobに合格 |
| metadata pass / binding / partition | 各1回、retry 0 |
| 回路build / state・Born / RNG・shot / compile / variance・N・費用 | すべて0 |

これは保存selected operationと共通factorizationの**literal十分条件**についての局所結果である。一般のoperator等価性、異なるbasisの代数的等価性、別slot/roundを探索していない。すべて単独群という結果から、案M一般やO2全体の不可能性を主張しない。既知ancilla-free/groupingへの優位性・分散減少・必要shot数や期待費用改善も取得していない。

## Identity・証拠

[SOURCE.json](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/SOURCE.json)に実行SOURCE・関連18ファイルのSHA256・環境、[INPUTS.json](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/INPUTS.json)に固定N1の5入力blob/SHA256がある。成功後にinput/sourceを修正していない。

| 保存identity | 値 |
|---|---|
| H | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` |
| partition | `900342ad6262021ff9b919c36ee182a8ed83914a697524e3fae07248d0d082f8` |
| preparation | `f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299` |
| tail | `890442900e605a262201338f19afdad638302d670c75a9d95371c10b83ea38d8` |
| selected policy | `f6eba8633890dbf09b5388928a369650251bf53567d51ab4179282c7c239b2cc` |

保存した[HEADER_BINDINGS.json](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/HEADER_BINDINGS.json)は各labelの適格/zero分類、selected/original basis、signed support、角/phase、wireとA/B/Vのliteral記述を保持する。[GROUPS.json](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/GROUPS.json)と[RESULT.json](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/RESULT.json)は群・質量・q・判定・UNKNOWNを保存する。n=0でのV=保存selected basisのadjoint、A/B application順、signed rotation、scalar/event phaseは固定binder規則のまま取得した。新しいunitary matrixでのtarget等価性認証を行ったものではない。

[開始記録](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/CALL_START.json)、[最終call数](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/CALL_COUNTS_FINAL.json)、[STOP.json](../../../artifacts/o2_m_n0_header/2026-11-08/metadata_run01/STOP.json)を保全した。初期CALL_COUNTSは0のままの開始時snapshotで、実際のattemptはFINALを参照する。取得済みJSONを再生成/丸め直し/再分類していない。結果報告のlabel/群件数は新出力の構造を表示したもので、追加science passや旧rawの数値再解析ではない。

## UNKNOWN・次の判断

未取得は、元gate表現が異なる場合のoperator等価性、実state上の相関/分散、M instrumentの数値誤差、hardware feed-forward、full compiled価格、必要N、全PR＋RPE費用である。SOURCE結果でもこれらはUNKNOWNとされている。N1/N2再生成、CDF再計算、旧engine認証、科学測定を自動起動しない。

今回のslotでは群平均の非自明な適用先が得られていないため、量子pilotへ進む根拠は得られなかった。別の同値判定・slot・手法に進む場合は、その技術的差分と有限scopeの新しい指示/実行許可が必要。今回、別slotや案D/O1/O3へ移行せず、重要科学レビューや研究方針の正式採択も行わない。

## 保全・公開

正式共通指示2026-10-25、適用AGENTS、PROJECT_MAPと既存実行契約を確認し、既存runnerだけを使った。入力前の共有サーバー負荷は低く、MemAvailableは約1 TB。新しいCPU/RAM/worker上限、共有設定変更、manager/grantは設けていない。正確なprocess別wall/RSSはこの固定runnerが保存していないためUNKNOWNとし、後から科学callを再実行して測らない。

元main・dirty/untracked、旧SOURCE・raw・結果・STOP・grant/ledger/marker・C1凍結、他worktree/process/cacheを保全する。科学SOURCEは旧commitに固定し、この結果branchでは新出力と報告/索引/manifestだけを追加する。commit/公開後、remote SHA・全公開本文・blob/tree・SOURCE/input byte不変・元main保全を照合する。公開結果SHAは最終チャットで示す。

公開前の技術照合はPASS。実行SOURCE18ファイルとN1入力5ファイルは固定byteのまま、生成した9出力も取得後のbyteを維持している。manifest旧21項目不変、既存Markdown7点は追記のみ、REPORTの10参照を確認。元mainのHEAD/statusと2,136ファイルを保全した。新しい科学call/既存testは追加実行していない。

**一回実行と結果保全を終え、科学的処理はmandatory STOP。公開照合後も追加計算・retry・条件変更・別slot探索へ進まない。**

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/39d839063c53d949342af4a0541b4865bdd054ed
[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39d839063c53d949342af4a0541b4865bdd054ed/docs/research/o2_m_n0_header_binding_preparation_20261108/M_N0_METADATA_EXECUTION_DRAFT.md
