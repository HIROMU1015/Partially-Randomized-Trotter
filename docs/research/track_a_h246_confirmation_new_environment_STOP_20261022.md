# H2/H4/H6高次PF PR：新環境の必須資料欠落による実行前STOP

2026-10-22 JST。**実行承認受領済み・科学実行前mandatory STOP**。
基準commitは`c7c1d8229508253390212588ae2b25e504eb4915`、科学SOURCEは
`8825391ff73b531be5834bf852b9bf2479d1799b`。
[元の固定契約](track_a_h246_higher_pf_confirmation_readiness_20261021.md)に従う
118 trajectory・236 compile軸の独立確認と取得後の固定評価は承認されている。
停止理由は追加承認の不足ではなく、保存科学入力とprivate証拠を現在利用できる保存先から復元できないことである。

## 確認できたこと

- GitHubから基準commitをfetchし、専用branch
  `h246-confirmation-new-environment-20261022`を作成した。
- 重要SOURCE68ファイルは指定科学SOURCEのGit blobと全byte一致。
  H2 adapter3ファイルも元SOURCE`4ec26921c6872ed6e8667f167074e1e405cf1a3f`と一致した。
  凍結済みshot・設計入力の評価JSONは期待SHA
  `b6d1309fc615c6df2c3a65c81f0a62dfff327435e98fde3bc2d74f3b80504f5e`と一致した。
- 既存環境はPython3.12.3、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、SymEngine0.13.0。
  compiler契約は`rz/sx/x/cx`、opt1、seed17のままである。
  private packがないため、旧runtime closure全体のbyte同一性は未確認。
- 既存の人工・mock検証は29 PASS、2 SKIP、15 subtest PASS。
  SKIPは`M5_TEST_PROVIDER`未指定のnative/compile人工テスト2件である。
  processpoolの成功・故障・回収、one-shot保護、固定予算、paired共分散のmockを含む。
  別担当の独立レビューやimmutable CIではない。

## 所在確認と不足資料

本人のproject・worktree・archive・data領域を、Git除外ファイルも含めて名前検索した。
本人所有の関連一時ディレクトリと、利用可能なmountの上位も確認した。
`/mnt/external_hdd`と`/mnt/nas_backup`は空の未mountディレクトリだった。
他ユーザーの研究データは開いていない。全host・全旧rawの網羅調査ではない。

対象project内の保存NPZ148ファイル（worktreeの複製を含む）と、既存H4移行tarの
NPZ6メンバーをbyte SHAで照合したが、以下の固定snapshotに一致するものは0件だった。
既存H6 exporter ZIPはツール・説明4ファイルであり、保存配列を含まない。

| 対象 | 必要な保存snapshotのSHA-256 |
|---|---|
| H2 | `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa` |
| H4 | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| H6 | `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005` |

必要なモデルはlinear H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8、保存DF rank3/10/19、
prefix1/3/5、T=.8、P=0、K4、ABS_TIME_PER_OUTER=32である。
旧H4 DF12や別系列の配列は代用しない。SCF・DF・状態・referenceを再生成していない。

そのほか、次の原資料が必要である。

1. private `proposal02`の3stage分の`config_DRAFT_PRIVATE.json`、
   `plan_DRAFT_PRIVATE.json`、元のinput/runtime/origin-config seals。
2. 各stageの`registry_PROPOSED_PRIVATE.json`、`excluded_seeds_PRIVATE.json`と、
   その根拠となる既知cohortのregistry・除外資料。
3. 固定snapshotに対応する保存state・reference配列、input/reference receipt、
   保存H2 signal、および契約に結合された元取得config・終了監査・completion・native exit資料。

SOURCEから新piece seed metadata846個（264/299/283、stage間衝突0）を決定論的に
再構成できることまでは確認した。しかし、元private registryのbyte同一性と、
約25.65万件の既知seed除外との非衝突は検証できない。
空の除外集合を正式資料として補充したり、公開plan/config digestから元のprivate closureを
推測で作ったりしていない。seed値そのものは公開成果物へ載せない。

## STOP・保全・次の条件

新実分子trajectory、sampling、simulation、compile、signal、reference、固定比較評価はすべて0。
実科学grant発行・ledger作成・one-shot marker作成も0であり、科学stageは未起動。
改善率・統計的不確かさは未取得のためnullである。過去の探索結果を独立確認結果へ転記しない。

既存SOURCE・raw・grant・ledger・marker・STOPへ書き込んでいない。
新しいSTOPと所在確認記録を別領域へ保存した。共有設定・他jobの変更、GitHub pushは0。
CPU/RAM/diskは読取観測のみで、科学入力gateが不成立のため実行資源のadmissionは行っていない。

必要なのは上記原資料の復元可能なコピーまたは保存先である。実行承認は受領済みである。
復元後も重要SHA・元契約・seed除外を確認し、未使用stageの新規grantのみを使う。
科学call消費済みstageのretry/resumeは新grantでも自動実施しない。
今回はこの実行前STOPをローカルcommitし、停止する。

機械可読記録：
[STOP](../../artifacts/resource_applicability/track_a_h246_confirmation_new_environment/2026-10-22/STOP.json)、
[不足資料](../../artifacts/resource_applicability/track_a_h246_confirmation_new_environment/2026-10-22/missing_inputs.json)、
[所在確認](../../artifacts/resource_applicability/track_a_h246_confirmation_new_environment/2026-10-22/input_search_summary.json)、
[SOURCE照合](../../artifacts/resource_applicability/track_a_h246_confirmation_new_environment/2026-10-22/source_checks.json)、
[承認範囲](../../artifacts/resource_applicability/track_a_h246_confirmation_new_environment/2026-10-22/authorization_scope.json)、
[人工検証](../../artifacts/resource_applicability/track_a_h246_confirmation_new_environment/2026-10-22/technical_validation.json)。
