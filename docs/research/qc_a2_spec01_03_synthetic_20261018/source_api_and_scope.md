# SPEC-01〜03 SOURCE/APIと保存形式

2026-10-18 JST。[技術報告](technical_report.md)／[handoff](decision_handoff.md)。固定仕様[7af83266…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_a2_certificate_connection_design_20261018/certificate_connection_spec.md)を実装した限定namespaceである。

## SOURCEとpublic API

| 新規ファイル | 役割・API |
|---|---|
| [common.py](../../../src/trottercert/a2_spec01_03/common.py) | Int/Nat/Rat/ComplexRat、canonical bytes、strict JSON、namespace付きSHA256、ObjectRef/MatrixRef、失敗record。SOURCEの自己hash読取以外の科学入力I/Oなし |
| [raw.py](../../../src/trottercert/a2_spec01_03/raw.py) | 明示full DFの人工raw schema/array descriptorを検査。scientific decode/cap/shiftは持たない共有基盤 |
| [generator.py](../../../src/trottercert/a2_spec01_03/generator.py) | `decode_binary64(data:bytes,endian:str)→Fraction`、`intake_model(raw)→record`、`generate_spec01_03(raw,particle_number:str,fragment_order:list[str]またはNone)→record` |
| [checker.py](../../../src/trottercert/a2_spec01_03/checker.py) | `check_spec01_03(raw,evidence,expected_particle_number:str,expected_fragment_order:list[str]またはNone,expected_model_id:strまたはNone)→record`。generatorを呼ばず元bitsから検算 |
| [__init__.py](../../../src/trottercert/a2_spec01_03/__init__.py) | 上記の公開と`canonical_bytes/content_id/loads_strict/read_rat`。SPEC-04以降のAPIなし |
| [人工test SOURCE](../../../tests/test_a2_spec01_03_synthetic.py) | literal/seeded人工raw、hand oracle、positive/negative/property/mutation/independence tests。実入力fixtureなし |

以下は呼出規約の説明であり、新しい科学実行runnerではない。実分子保存入力への適用は今回承認されていない。

`intake_model`はraw dictまたはJSON textを受け、`INTAKE_ACCEPTED_EXACT_NOT_CERTIFICATE`か理由付き拒否を返す。`generate_spec01_03`は同じrawと期待N/全fragment順を受け、`EVIDENCE_GENERATED_UNVERIFIED`と`evidence`を返す。成功flagをcheckerが信頼することはない。checkerへはproducer wrapperでなくevidence objectを渡す。

checkerの受理verdictは`SPEC01_03_VERIFIED_EXACT_SYNTHETIC`。受理範囲はexact model bits、N文法、PSD分割、L、f_j、Λ、S2/scalarだけであり、必ず`not_verified`にU/W/K_SOS/Cbar2/step/state/回路/native/元分子対応を含める。`expected_particle_number`は必須、PF順の既定はone_bodyの後にraw全block順である。別の正当な全順序を使う場合はgenerator/checker両方へ明示する。

呼出側が外部固定model_idを持つ場合、checkerの`expected_model_id`へ渡す。入力自体を変更して正しい別証拠を作り直した攻撃と、単なる数学的に別の正当な入力を、trusted contextなしに区別できるとはしない。元データ・完全inventoryはprovenance/別の固定hashから確認する責任が残る。

## Raw envelopeとexact model

raw schema/versionは`A2_RAW_DF_V1`、意味論は`DFHAMILTONIAN_FULL_SIGNED_SQUARES_EXACT_BINARY64_V1`。全keyはstrict、n_modes/N/shape等の整数は十進string、科学量はRatのnum/den stringで保存する。raw bitsは小文字hexであり、JSON数値を一切必要としない。

必須fieldは`schema,semantics,n_modes,mode_order,conventions,units,block_inventory,arrays,provenance`。arraysはfull constant、one_body、lambdas、各block_id付きg_blocks。各array descriptorは`shape,axes,dtype,endian,order,unit,data_hex`で、C order、明示little/big endian、float64/complex128のみ。c/λはfloat64、q/gは実または複素。複素arrayは要素ごとに実部・虚部のbinary64を並べる。

mode0は整数bitのLSB、occupation basisはcreation index昇順、onebodyはa†_p a_q。q/gのaxesはmode_orderと一致し、λ/g_blocksの順序はblock_inventoryと一致する。callerがbit順・外部API規約を暗黙変換してはならない。固定Nの0端点・満占有、形式的な0-modeも人工testで検査している。零λ/零capはinventoryへ保持する。

provenanceは人工input class、full constant/onebodyを一度使用、truncationはmetadata-only、full block inventory、元分子未認証、transform_history/source_refs/metadataを持つ。source_refsはbytes hashまたは固定Git ref型を検査するが、その外部ファイルの存在や科学的妥当性をネットワーク照会しない。full/correction-onlyの別形式はこの初期versionでは拒否し、暗黙adapterを持たない。

固定DFモデルの係数はbinary64 bitsの正確な有理数。元分子の丸め/truncation/model差をゼロとみなした意味ではない。原分子未認証はprovenanceに残し、科学演算は発動しない。

## Objectとhash graph

各nodeは`{id:SHA256,payload:object}`。namespace付きcanonical JSONからdigestを作り、自己idをpayloadに含めない。

- `intake`：canonical raw envelope digest、exact model node、array refs、provenance。
- `model`：full係数、mode/block順、単位/CAR/意味論。raw bits endian・provenanceはmodel hashに混入せず別ref。
- `sector`：model_id、固定N、mode/CAR規約、number-conserving文法。
- `cap_witness`：model_id、entrywise method、exact s_q/u、対応するexact coefficient IDs。
- `partition`：model/sector IDs、cap method/値、ordered fragments、L。
- `pf`：partition ID、全fragment順、display/time stages、scalar台帳・phase規約。
- `LowerCertificate` / `FragmentCapCertificate`：model/sector/partition/PFの親IDs、raw_input_id、witness ref、exact値/fragment/由来、SOURCE version。
- `evidence`：SPEC-01〜03限定completion、SOURCE hashes、全context/上記nodes。**bundle/comparison certificateではない。**

hashの一致だけで数理受理しない。checkerはraw→exact model、cap、signed base/PSD/shift、L/scalar、PF、全certificateを再構成し、最後にSOURCEと全node hashを照合する。SOURCE identityはこのpackageの5 moduleのbytes hashを持ち、repositoryの固定commitから別に追跡する。証拠形式が合うこととcheckerが数学的検証を受理したことを分ける。

## 拒否とpartial record

`INVALID_MODEL_FOR_GUARANTEE`、`GENERATION_REJECTED`、`IDENTITY_MISMATCH_INCOMPARABLE`、`INVALID_CERTIFICATE`、`INCOMPLETE_CERTIFICATE`、`TECHNICAL_ERROR_OR_INTERRUPTED`をstage/reason付きで返す。失敗の`verified_quantities`は空。入力がcanonicalならvalidation前の提出digestを保存し、非canonicalならhash不能の理由を残す。例外をapproximate repairへ転換しない。

returned recordsはcallerがファイルへ保存できる。API自身に任意pathの読み書きやjob/marker発行機能を付けていない。最終的な科学実行の保存/STOP/authorizationは未実装である。

## 人工testの再現

承認済みの人工testを単独で再現するコマンドは次である。既存pytest conftestを避け、今回のtestファイル以外を収集しない。

```bash
PYTHONPATH=src python -B tests/test_a2_spec01_03_synthetic.py
```

Python≥3.11、stdlibのみ。外部パッケージの追加・共有venv変更は不要。stdout/logが人工技術testであることを維持し、今後の実入力検証許可と混同しない。今回の記録付き実行は同じunittest suiteをTextTestResultで収集したものであり、科学runnerの実装ではない。

**Mandatory STOP。**
