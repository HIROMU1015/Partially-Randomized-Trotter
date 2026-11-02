# H2 PR資源評価：Phase 0静的監査とF承認待ちseal

2026-10-19（Asia/Tokyo）。**F_SEALED_PENDING_EXPLICIT_APPROVAL。科学実行は未承認・実行0。** ユーザーの「これがその新規チャットです」により、このチャットをH4検証から独立したH2専用チャットとして使用する。新規チャットを再度要求しない。

## 固定SOURCEと今回の到達点

- 科学SOURCE：`37b1cf72990885962d8449d8bd5ec7e303329350`
- H2 branch：`codex/h2-pr-resource-preparation-20261019`
- 準備成果物：[phase0_preparation01](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/phase0_preparation01/)
- 契約要約：[F_contract_summary.json](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/phase0_preparation01/F_contract_summary.json)
- 私有の承認待ちspec：`.h2_pr_resource_private/2026-10-19/phase0_preparation01/F_spec_SEALED_PENDING_APPROVAL.json`
- spec digest：`bc78b7e2bb759181aa52d0202483f0916842689c6731da437ad008e8848a9da7`

今回完了したのは、既存H2の静的監査、FのH2次元adapter、承認gate、one-shot/native wait4の実装、人工検証、FのSOURCE・生成規則・予算・runtimeの固定である。H2のHamiltonian・状態・実rankはまだ取得していない。Phase 0の科学的検証完了、D/A最良、S_STATSのn、P最終sealを認定しない。

## 既存H2データの再利用監査

固定基点 `d728d088c010089c6b672eef0521261791b2d209` の追跡済みH2 artifact 52件を、file bytes・SHAと保存metadataで整理した。今回の基点とresource_applicability配下で、今回のtol-only入力と同一性を確認できるH2結果は見つからなかった。探索はこの固定範囲に限定した。未追跡・私有H2記録が他の場所に存在しないとは主張せず、共有projects・/tmp・他チャット履歴の広域走査は行っていない。

既存のH2 PF v4／D6 JSONは4 qubits・rank3だが、保存metadataは `df_rank_source=config`、`df_tol_requested=null`、projectの3/4 rank選択である。保存sector次元は6であり、今回のNα=Nβ=1のspin sector次元4とも異なる。tol-onlyと同じrankになる可能性だけでは入力同一性を認定できず、これらのHamiltonian/state・誤差係数・PF資源を今回の主評価へ流用しない。旧係数・Pauli-tailテストは歴史的文脈／実装参照として保持し、実分子を作る旧H2テストの再実行もしない。

[監査一覧](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/phase0_preparation01/existing_H2_inventory.json)。旧結果のSOURCE記載は保存来歴として転記したもので、dirty-worktree結果の全SOURCE同一性を新たに認定していない。

## Fの生成条件とcall予算

linear H2、座標(0,0,0)/(0,0,1) Å、STO-3G、charge0・multiplicity1、RHF、SCF conv_tol=1e-9・max_cycle50、symmetryなし。H4の固定Fと同じSCF／積分helperを使い、canonical MOの追加sort・位相変更を行わない。二電子積分のspinorbital変換と1/2係数を維持する。

DFはOpenFermion `low_rank_two_body_decomposition(..., truncation_threshold=1e-8)`だけを指定し、final_rankを渡さない。実rankは返却値から記録する。生成順・signed係数・全返却fragment・後続cutoff0を維持する。weighted Hermitian projectionはfull-Fock N=4に合わせて同じ式を使用し、既存9.9e-11 Hartreeのengineering gateを維持する。これは区間保証ではない。

状態はNα=Nβ=1、sector次元4の最低 `scipy.linalg.eigh` vector。最大振幅を実正にする位相規約、保存後再正規化なし。独立occupation oracleとの行列差<1e-9、Hermiticity<1e-10、状態残差<1e-10、正規化誤差<1e-12を確認する。最低固有値gap<=1e-12なら、縮退時の別方針を勝手に選ばず証拠を保存してSTOPする。ground-state certificationは付けない。

| 科学callのカテゴリー | Fの固定予算 |
|---|---:|
| SCF kernel | 1 |
| integral build | 1 |
| tol-only DF decomposition | 1 |
| sector行列のDF作用column | 4 |
| 独立occupation行列の構成 | 1（4 column） |
| dense eigensolver | 1 |
| signal / sampling / wrapper compile / shots | 各0 |

integrals・raw DF・最終snapshotとreceiptをH2専用attempt内に保存する。上記は高位科学callの予算であり、SCFの内部反復回数やBLAS呼出し数を1と数える意味ではない。SCF未収束などの失敗を理由に再kernel・自動retry・別stageを開始しない。

Fは1つの小さい入力生成であり、並列な科学候補はない。4×4 sector作用にはPython backendを使用し、不要なJITを避ける。CPU/RAM/worker/wallの歴史的上限や新しい恣意的上限を設けず、実行時のaffinity・空きRAM・PSIを観測する。PSIや無関係なworktree更新だけで正常な科学workerを強制終了する契約にはしない。

## 再利用と非干渉

H4 F `scripts/h4_tol_only_input.py` の生成・snapshot・projection・loadを基に、H2の軌道／sector次元、専用契約、予約台帳、縮退判定に必要な差分だけを `scripts/h2_tol_only_input.py` に実装した。JSON/digest/exclusive書込helperは既存 `h6_dba_contract.py` を再利用する。

providerは固定SOURCE `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` のsrc 262ファイルを、H2私有領域へ通常ファイルとしてコピーした。H4 worktreeへのsymlinkや実行時の可変H4 SOURCE参照はない。独立occupation oracle・DF Hamiltonian・状態hash／phase規約もこの固定providerを使う。

Pythonは既存のproject専用native-dependency-v2 runtimeを読取利用する。共通環境を変更していない。既存の固定runtime一覧から、実行コード・共有ライブラリ等12,202ファイルをH2用manifestへ固定し、該当する旧hashとの不一致0を確認した。非実行のheader/static archive/test fixtureと可変pycは除外し、`-B -X pycache_prefix=<H2 private>/runtime_pycache` を必須化して旧pycを読まない。TMPDIR・Numba・Matplotlib・XDG cacheはH2 attemptのscratchへ向ける。

H4/H6/H8のSOURCE・契約・入力・grant・ledger・marker・STOP・結果・cohort・実行プロセスを変更・混合する操作は行っていない。それらの現時点の実行状態を再監査したとも主張しない。関係のないHEAD/status変化は凍結gateにしない。

## 人工検証と承認境界

人工16件PASS。H2 shape、signed projection、過剰projection拒否、snapshot roundtrip、object拒否、人工の分子APIだけを通したF全経路／call一致、明示承認・scope・SOURCE/provider/runtime／receipt改変拒否、予約の再利用拒否を確認した。実分子を使用していない。さらに、今回の実際の承認待ちspecとINVALID grantをdriver／worker／direct generateの3入口で拒否し、実科学package import・attempt作成前に停止した。

[人工検証](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/phase0_preparation01/artificial_test_report.json)、[3入口拒否](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/phase0_preparation01/refusal_report.json)。作者自身の検査であり、独立科学レビューではない。

科学stageの有効grant、科学attempt、ledger、one-shot marker、新seedは0。人工fixtureのファイル・偽grantは専用一時領域だけで使用し、実科学承認として扱っていない。後段FP/D/A/S_STATS/Pの依存関係草案は保存したが、SOURCE・実rank・候補・n・seed・call予算は未sealである。H4のrank10/D190/A50/n1108/78は使っていない。

## 次の承認とSTOP

**ユーザー**：上記SOURCE、生成条件、固定call予算のH2 Fを1回だけ実行するか判断する。今回はFとFPを分けた具体的提案として提示し、FPを承認に含めない。

**Codex（Fの明示承認後）**：元の承認待ちsealを保全し、ユーザー承認receiptを結合した新execution spec／grantをH2側に作る。直前のSOURCE/input/runtimeと実空き資源を確認し、Fのみone-shot実行、native exit・同一親wait4・最終bytesを監査してmandatory STOPする。

F後に実rank・保存入力を確認し、H2で意味のあるFP/D/A集合を事前登録する。適格候補なしのP適用外扱い、S_STATS提案nの最終採用手順は未採択の補足事項として維持する。P sampling/compile、詳細科学レビュー、論文最終evidence freezeへ進まない。pushも未実施。現在はF承認待ちのmandatory STOP。
