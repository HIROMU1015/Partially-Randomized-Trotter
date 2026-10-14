# 一体項再配分PF：固定一回pilotの実行承認用引継ぎ

2026-10-15 JST。[実装・比較条件・費用の全仕様](qc_onebody_repartition_finite_step_preparation_v1_20261015.md)。完了済みGPTレビューに基づく準備で、新しい科学レビューは行っていない。**現在は数値実行未承認・Mandatory STOP。**

## 1. 承認対象

scope `QC_ONEBODY_REPARTITION_FINITE_STEP_V1_20261015`、branch `qc-onebody-repartition-finite-step-preparation-v1-20261015`、source `19d5bd74db020564ea9a4b1882e102911e983cdb`。[固定JSON契約](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json)、[source/input identity](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/source_manifest.json)。

既存linear H4、距離1.00 Å、STO-3G、DF rank12保存pack、8spin/4spatial、Nα=Nβ=2、36dim。Hamiltonianは保存de7a、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。cached ground stateを使い、状態の再準備・分子生成はない。scalar/normal-order/rank/order/sectorは変更しない。新規Hermitian truth-H diagonalizationはcached state検証/真値評価用に1回だけ。

最初densityのspin-paired diagonal X、projected-gradient1方向、最大6 leading候補をtruthなしで選ぶ。3構成×S2/S4×6h=36PF cells。各S2 scalar補正と全構成の隣接二点外挿を加えた69method records、target10^-4/10^-5 Ha・2価格で276cost rows。original subdivision、corrected original、S4、既知linear extraction、Richardsonを含む強対照の最小eligible費用へ戻す。

hはcommon coefficient boundから固定formula `1/(8Λmax2^s),s0…5`。実行後の入力・direction・h・閾値の救済追加はない。10% material effectを全4contextに要求するが、判定は条件付きcontrolled primitive/Heisenberg価格モデルのcomponent gainに限る。precision grid選択は事後oracle、情報取得/compile/T/全QPE費用は未確定で、有限窓の結果を研究正式採択や新算法認定にしない。

## 2. 一回の上限と独立監査

- pack load1、分子生成0、ground-state preparation0、spatial eigh≤15、truth-H eigh≤1、PF Schur≤36。
- producer RDM contraction1、direct leading≤3、PF seeds≤36。
- independent JW/Pade/direct checker：PF再合成≤36、expm≤1024、RDM収縮1、direct leading≤3、新eigensolver0。
- compile/T合成/iterative optimization0、1worker/1thread、RSS512MiB、output32MiB、log2MiB、wall/CPU capなし。
- attempt1/retry0。認可・source・environment・入力照合後、数値import前に一回markerを作る。failure時も消費済みで停止する。auditも一回。

source preflightと37人工testsは通過しているが、今回のreal packに対するH保存・actual density noncommutation・ground residual・γ/d0・finite-h error/overlapは未取得。これらは上の認可範囲に入る新しい科学計算であり、準備PASSを実行承認へ代用しない。

## 3. 実行承認の境界

[authorization_target.json](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/authorization_target.json)はUNISSUED/approved=false。現在の `--execute` は拒否する。ユーザーの別明示承認後に、targetを書き換えず、同source/input/contract hashへbindした別のissued approval記録を作る。旧C1 940-callや事後diagnostic認可は流用できない。

承認する場合の範囲を明確にする指示例：

> 一体項再配分PFの固定H4一入力・36PF評価セルと、固定上限内のRDM/leading/truth検証・独立監査を、GitHubのsource19d5bd74・固定契約どおり一回だけ実行することを承認します。compile0、attempt1、retry0とし、結果・失敗記録のGitHub公開と再取得確認後にMandatory STOPしてください。

この例文は承認そのものではない。現回答や受領レビュー、source/testの公開から自動発行しない。

認可後の操作はrunner `--execute --authorization <issued record>`、続いて独立checkerの固定attempt directoryへの一回監査。BLAS/OpenMP/Numba thread1はsource gateで必須。詳細commandとsource/contract/input SHAはtarget/planの固定値を使う。

## 4. 得られる情報と得られない情報

得られるのは、固定input/selector/gridでH保存とresponseが実fermionic表現へ接続するか、同じXのfinite-step bias/overlapがどう変わるか、強い補正PFを戻した条件付きcontrolled component費用差があるかである。positiveの差額は追加情報取得の価格予算として提示できる。

得られないのは、最適h/回路、truthなしprecision選択、有限shotのRDM取得法、T/depth/全confidence費用、実分子移送、独立新規性、全QPE/RPE総資源の優位である。leadingだけの改善は成功としない。bias floor/branch/numerical/eligible不足はUNDETERMINED、negativeも固定scopeの結果として保存し、追加scanで救済しない。

重要な新結果の継続/縮小/終了/第二候補への移行判断は、結果取得後に別開始承認を受けたGPTへ戻す。現在は同じ設計の再レビュー不要。N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、Track A/B、C1–C4、既存FAIL・認可・STOPを維持する。
