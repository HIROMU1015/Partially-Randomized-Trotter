# Signed support実H4 metadata-only：契約具体化の報告

2026-11-07 JST / SERVER1（GPUA100-gleap-01）。**契約草案の静的設計を完了。実H4の新feature/moment/非定数性評価、科学runner実装、新テスト実行は0。**

## 結論と最小接続

b23b3c0の既存signed kernelをそのまま再利用できる接続仕様を具体化した。N1 `744e1a09661ebba6841fe5faf456f88c41c47780` の **LAW.jsonとPREPARATION.json全bytes** が直接入力であり、catalogは後者のcomponentsである。commit・relative path・byte SHA256・Git blob・H/preparation・固定a,b・qubits・feature versionを[固定binding](SIGNED_SUPPORT_H4_METADATA_BINDING.md)に記載した。元blobとcheckoutのbyte一致を確認した。

次回は保存2入力を各1回読み、既存metadata_from_bytesでbindingし、SignedControlを1回だけ構築する。そのexact ν/m₂/Σを全5層について保存し、Σ1>0かを一度判断してSTOPする。state・engineは参照commitだけで、数値ロードしない。Born・shot・compileは不要である。

## 既存SOURCEの再利用と最小差分

sampler/law、scalar binder、identity・5層検査、SignedControlの有限和は既存のまま。新しいsampler/OLS/framework/managerは不要。必要な次回差分は、固定2入力のI/O、外側の保存5層/label順の照合、call記録、exact結果のatomic保存、STOPをつなぐ薄い入口だけ。今回はその実装を行っていない。

既存basisは `560c6092bfb5e71af8eedc4a4be5416922764f98`、signed基準は `b23b3c059dd1fdcc86178659311983048dea2257`。実入力投入前に、次回追加入口を含む新しい実行SOURCE SHAと環境を固定・公開照合する必要がある。既報106人工PASSは再利用の根拠として参照し、実H4接続成功の証拠とはしない。

## 有限scope・適格性・STOP

[実行契約草案](SIGNED_SUPPORT_H4_METADATA_EXECUTION_DRAFT.md)は、LAW/catalog各1読込、signed binder1、SignedControl1、全5層moment1組、h1非定数性判定1、science retry0を規定する。内部のJSON解析、from_record、正規化/grid/positive support、5層inclusion-exclusionによる照合を明記し、「全内部演算が一回」とは扱わない。新CDFや全path列挙は不要。

全検査PASSかつexact Σ1>0なら構造的ELIGIBLE、Σ1=0ならINELIGIBLE_CONSTANT_H1、identity/schema/moment/call等の不整合はUNKNOWN_ERRORとして取得済み証拠を保全し、いずれもmandatory STOPする。成功してもtrain/pairedへ進まない。X/Y相関・finite trainのrank・残差分散・学習費回収・RZ改善はUNKNOWNのまま。

出力はSOURCE/input identity、signed metadata/control identity、exact全5層値、適格性と理由、call/例外記録、REPORT/STOPを保存する仕様。今回これらの実H4数値を生成していない。

## 未決定事項と次の担当

- **ユーザー**：このmetadata-only有限契約と、次回の最小入口準備・公開固定後の一回実行を採択/明示許可する。今回は科学実行許可ではない。
- **SERVER1 Codex**：許可後に既存kernelを変更しない薄い入口と人工検査を準備し、SOURCE SHA・最終fresh output・環境を公開固定して限定実行する。
- shot数・seed/stream・train/eval配分・range guard・価格条件はmetadata-onlyには不要で、今回未設定。fresh学習/paired評価は適格性結果を踏まえた別契約・別承認である。

既存224結果 `ba483ab3038693048e58830e39a676493a892053` はfragment indicatorの完了結果で、signed supportの判定ではない。方向整理 `4741cae05af662dfc42d0519ff977756a94cb7b7` とconfidence接続 `af47cf9161241c8710383ff8539a4a7f8f67492c` の未検証/STOPを継承する。旧bit・係数・価格・cacheの再解析なし。752結果 `3c8e55a2728f86a1eda17a2e0105050a637e7c86`、N2保証state、engine認証、C1凍結を変更しない。

## 原本保全・公開・mandatory STOP

独立worktree `/tmp/scv-signed-support-h4-metadata-contract-20261107`、専用branch `codex/scv-signed-support-h4-metadata-contract-20261107`、base b23b3c0を使用する。今回追加するのは本directoryの指定3 Markdownだけ。元main・dirty/untracked・固定SOURCE・結果・grant/ledger/marker/STOPを保全し、O1/O2高速化セッションのSOURCE/process/cache/成果物は調査・操作しない。

正式共通指示§8.1/8.2/8.4/10/10.3、適用AGENTS、PROJECT_MAP、研究概要・現状と関連固定契約を確認した。今回の「3文書のみ」という範囲に従い、旧索引・manifest・科学SOURCEを変更しない。新しい承認証明・grant発行・manager・重要科学レビューはない。

公開後にremote SHA、3文書の本文・blob/tree、旧SOURCE/結果の無変更を照合し、**mandatory STOP**する。metadata数値評価、科学runner実装、train/paired/calibrationへ自動進行しない。

公開前のbyte照合では、既存SOURCE/索引2,923ファイル、元main既存3,087ファイル、旧停止研究1,086ファイル、前pilot29ファイル、SCV224 live raw530ファイルとsigned新SOURCEの無変更を確認した。元main HEAD/statusも一致。保全検査はbyte hashだけで、保存bit等の科学的な再解析を含まない。追加差分は指定3 Markdownのみである。
