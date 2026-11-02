# GPUA100-gleap02向け非機密handoff

2026-10-23 JST。GPUA100-gleap-01での原本byte確認を伝える限定projection。新成果物はローカルbranch `codex/h2-common-pf-original-recovery-20261023` にだけcommitする。公開承認・push・私的payload転送は未実施。受信前に「GPU02で原本byte確認済み」へ変更しない。

1. [SOURCE.json](SOURCE.json)でaudit_SOURCEと再利用SOURCEを固定する。[fixed_sources.json](fixed_sources.json)のcommit/path/SHA/bytes/blob OIDを使い、既存資料を再登録せず参照する。Bは `ca29dd9ddc8a09cace84310f0dfdf7124652e07a`、保存PFは `f36402f90bcfcbcc4d556c7a952837dc5d7dd291`、科学SOURCEは `1b35f9ac2efab4c15bc00ee2d3c81643aca422f9`。GPU01索引 `e0c942fbb5dd8741dda69af0933fce88e8eaea91` と3環境索引 `6dafed86cc82149d55b4e6395382953d5f3631d7` のPF_COMMON_H2参照も確認した。
2. [axis_identity_4.json](axis_identity_4.json)のcandidate/axisをkeyに、Bの `saved_common_PF_cost_objects_4.json` に結合する。候補はH2_FULL_3_4_2ndとH2_FULL_3_2_4thだけ。cohort/native/SOURCEの異なる旧1367/2033へ置換しない。保存1655/2463は単価の転記で、新u/N/Gや順位を計算しない。
3. [original_identity.json](original_identity.json)の原plan SHA、入力5件、原result SHA、preparation hashを結合する。`native_results.json.gz` は固定PF commitの `artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/` 以下。JSON pointerは4軸表にある。原resultのファイルSHAと、gzip内のcanonical packet digestを混同しない。
4. [archive_identity_12.json](archive_identity_12.json)でtoken化相対path→archive SHA/bytes→raw SHA/bytesを結合する。QPY8件はv13/Qiskit1.3.0を既存parserで全byte消費済み。vector4件はSHA確認のみ。raw/npzを科学実行・数値decodeしない。計算したGit blob SHA1はprivate原本のGitHub収録を意味しない。
5. [runtime_identity.json](runtime_identity.json)は原plan closureの365 entryを同じindexで保持する。private locator mapのclosure[index].pathへ一対一結合できる。公開projectionとのrole/SHA集合だけだった段階から、原planの宣言path bindingまで確認済み。ただし完全なhistorical実loadと全transitive依存のreceiptは未発見。Bの元PRとの30対象binary一致はBの固定資料を参照し、今回元PRを再探索しない。
6. guardはcompiled QPY metadata・原result・B公開表が同一。correctnessはsingle saved input stateのstrict <1e-7、phase_alignment=falseの過去receipt。all-state/operator、現在共通native全stream同等性、A独立diagonal、3.12対3.11出力同等性を新たに認定しない。対象PIDのexit0/reapedと旧STOPは固定receiptへ結合済み。

private原本の保持者はこのhostの本人所有研究領域の管理者。非公開で `locator_map_PRIVATE.json` と `transfer_manifest_PRIVATE.json` を受け取れば、再探索は不要。manifestには21 member（原plan、実行入力5、原result2、worker_exit1、対象archive12）の元pathと相対member名・SHA/bytesがある。授受経路・宛先はユーザー管理のアクセス制限付き経路で別途合意し、既存原本をread-onlyで受け渡す。planには内部pathがあるためGitHubへ載せない。grant/ledger/CONSUMED、他候補raw、seedは引渡しmemberに加えない。runtimeの絶対path mapは必要な担当間に限って非公開共有する。今回packet作成・転送は0。

新projectionはREPORT/SOURCE/4軸表/12 archive hash表/closure role・hash表/保全・STOPだけ。公開には別の公開承認が必要。private原本が届かなくても、今回GPU01が検証した範囲は静的に結合できるが、GPU02自身による原本byte再検証とは記載しない。

科学call0、compile0、simulation0、COMPARE0、正式採択0。今回の原本回収を次段階の科学承認へ読み替えず、mandatory STOP。
