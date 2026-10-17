# A2 Phase B：正式科学契約C

ユーザーの2026-10-18明示採択に基づき、Phase A草案を新しい [machine契約C](execution_contract_C.json) として固定する。[採択記録](adoption_record.json)、[発言抜粋](user_adoption_excerpt.md)、[identity台帳](identity_manifest.json)、[静的検査](static_inspection.json)、[引継ぎ](decision_handoff.md)、[証拠索引](evidence_index.json) を合わせて参照する。

**契約は採択済み、Phase B科学実行は未承認・未実施。** `adopted=true` と `contract_grants_execution=false` を分離した。A2研究自体は `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認・未実行を維持する。科学SOURCEを変更・再固定していない。旧草案・proposalの未採択表現は履歴として保存する。

## 採択条件

| 項目 | 固定条件 |
|---|---|
| 対象・モデル | H4／1.00 Å／STO-3G／保存DF rank12。保存binary64値を正確係数としたfull DF Hamiltonian `cI+Q+ΣλA²` |
| sector・mode | total N=4、保存順8 modes m0〜m7。化学的spin sectorの追加主張なし |
| DF inventory | 保存全12 blocks。値を見てrank・block・順序を変更しない |
| PF | one_body → block0〜11、既存正stage palindromic S2。表示積は左→右、作用時間順は右→左、`exp(-ihH)`。scalar ledgerを保持 |
| U用trial | occupied indices 0,1,2,3、CREATION_ASCENDING／MODE0_LSB、determinant上界専用。ground/HFとの同一性を主張しない |
| d | `{"num":"1","den":"1000"}` Ha |
| α_spec | `{"num":"1","den":"10"}`。既存保証のspectral parameterであり、新しい測定confidenceやground overlapの実測値ではない |
| sqrt_bits | `"96"`、既存dyadic上側sqrt witnessの解像度。全演算を96-bit floatに変更する意味ではない |
| root_width | `{"num":"1","den":"1099511627776"}` Ha^-1（2^-40） |
| max_bisections | `"256"`。幅未達は未解像として保存。自動増加・閾値緩和なし |
| caps・比較 | `ENTRYWISE_RE_IM_ABS_SUM_V1`、RF0/RF1/EW2対centered・grouped構造付きCbar2。全検証済み証拠を両routeに同じように共有し、取得費用を無料にしない |
| record・log上限 | `"1099511627776"` bytes／record、`"1073741824"` bytes／stdoutまたはstderr log。予約・使用目標・総使用量の保証ではない |
| CPU・RAM・wall | capは全てnull。workerは既存 `AVAILABLE_RESOURCES_NO_SHARED_CHANGES`。新しい人工的低上限なし |
| state | 条件付きexact ground access。state preparation／native費用／総資源改善は未評価 |

保証対象は固定DFだけであり、元電子Hamiltonianとの厳密な対応、電子積分・DF誤差、状態準備可能性、QPE/native総費用の認定を含まない。数学と実装の既存の適用域・checker受理条件・結果分類は変更しない。今回、実係数のfinite・exact Hermiticityやcertificateの実入力correctnessは検証していない。

## SOURCE・入力・実行場所

Phase A基点は `a0629ca65142644a38cef545a9ad1244e6387eec`、SOURCE Sは `8d483f9bfa5983bb52fef55887e2f9f73a790511`。Cは草案のsource objectを全31 file hashごと複製し、不変性を検査する。入力はcommit `34d9a2f1a25bd35c03566ce0635e8b66665ad996` の保存NPZ、3,589 bytes、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。Phase Aのmember/header/array manifestをそのまま採用し、今回はcontainer bytesの照合だけ行う。

raw/model/sector/partition/PF certificate IDは未生成。`DERIVE_EXACT_FROM_PINNED_RAW_AND_MAPPING` は未決値でなく採択済みの決定的な導出policyであり、認可された将来のexact intake/checkerで生成・確認する。identity台帳のmapping/order/trial/task hashは**契約メタデータのhash**であり、実入力certificate IDではない。

canonical execution checkoutは既存 `/tmp/prt-qc-a2-pilot-phase-a-20261018`、ledgerはその `artifacts/qc_a2_pilot_execution`。管理用checkoutと分離し、canonical checkoutを移動・変更していない。Cのraw bytes SHA256で固定されたclaim pathをread-only確認し、今回ledgerもclaimも作らない。採択時点のcheckout HEAD/index/SOURCE/input bytes/ledger非衝突を記録したが、実行直前にも再確認が必要である。

SOURCEでのguardはC採択・別authorization・外部contract/auth/SOURCE/input manifest pinsを全て必要とする。別run_id・別checkout・再serializationによって消費済みCを迂回しない。後日のauthorizationは**このCファイルのraw bytes SHA256**をbindし、canonical JSON hashと混同しない。契約を勝手に書き直して新しいclaim枠を作らない。

## 呼出し・保存・STOP

既存call inventoryを維持する。adapter、SPEC-01〜03 generator、checker、SPEC-04〜07 generator、checkerは各1回、後段内のprerequisite再検査は計2回。最初の失敗で後段を止め、独立checkerをgenerator flagで代用しない。事前guard拒否ならinput IO前・claimなし、guard後はinput IO前にexclusive claimを消費し、以後の失敗・中断も消費済みのまま保存する。自動retry・parameter変更・SPEC-08への移行は禁止、全結果でmandatory STOP。

元input、contract/auth提出raw bytes、exact raw、証拠提案・checker報告・partial理由・hash・段別CPU/wall/RSS・Rat bit長・実保存量・stdout/stderrを既存SOURCEに従って保存する。保存失敗やSIGKILLなどの限界はbest effortとして明示し、claimを削除しない。最終log/result書込み費用の計測外範囲も元SOURCEのまま表示する。

空きdisk・メモリは [環境観測](execution_store_observation.json) に記録する。1 TiBの事前予約やそれ以上の空き容量をadmission条件にしない。実行時は実際の不足・pressure・共有環境への非干渉を確認し、他者のjob・設定・共有venvを変更しない。巨大証拠がGitHub通常file制約を超える場合、原本を安全に保全しmanifest／SHA256／要約を公開する。サイズ上限を理由に証拠を黙って切り捨てず、記録に限界を示す。今回は科学出力を生成していない。

## 今回の確認範囲

SOURCEのAST、契約専用メタデータ検査、Git ancestry・file bytes/SHA256、実行guardの静的順序、旧成果保全、公開同一性を確認した。full preflightやrunner、adapter、数値generator/checkerを実行しない。Phase Aの134人工methods／6,777 subtests PASSは旧記録への参照であり、今回の再実施・実H4成功・immutable CI・外部再現と扱わない。

**完了後mandatory STOP。次の担当はユーザーであり、この固定CとSOURCEに対するPhase B one-shot科学実行を別途明示承認するか判断する。**
