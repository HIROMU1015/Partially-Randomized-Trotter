# 層別identity・canonical出力schema

本書と[B_SCHEMAS.json](B_SCHEMAS.json)はschema/specだけ。今回科学inputやそのdigestは作らない。

| identity | bindingする内容・digest |
|---|---|
| SOURCE | active10fileの実bytesとcommit、canonical file→SHA mapのdigest |
| Fixture | exact math/JW/register/state/scalarとS/P alternativesを含むinputs/physical.jsonのcanonical bytes |
| Catalog | 独立4label・frame参照・c_cat/contextを含むinputs/catalog.json |
| Arm | ordered atoms／origin／sign／coverage／L／lambdaを含むarms/{arm}.json |
| Law | F1またはcatalogのq/r/K、exact lambda、binary64 p/B/angle/API metadataを含むlaws/{name}.json |
| Recipe | policy SOURCE identityのみ固定。realized gate recipe price identityは未binding |
| Environment | private dedicated interpreter/wheel/package/native receiptとそのdigest。public summaryへ内部pathをコピーしない |
| Execution | final contract／call plan／run ID／fresh output／新authorizationとimmutable attempt records |

SOURCE manifest一つのhashで全層を代用しない。I tail_hashはatom viewのsemantic digestで、arm fileのenvelope bytes SHAとは区別する。管理用spec文書hashと、B後のfixture科学input hashも別である。

## Serializer

UTF-8、ensure_ascii=False、sort_keys=True、compact separators、allow_nan=False、scientific data record末尾newlineなし。SHA256は保存した正確なbytesに対するもの。content digestを自分自身へ埋め込まずmanifestで参照する。

exact fractionはI fraction()と同じ既約形。整数は"0"/"1"等、非整数は"-3/7"等。coefficient mapはword配列のlexicographic order、typed atomsはIのstable atom_id順。word配列はqubit0から。S由来の区別とP全体collectを保つ。exact zeroのみ削除、signはexact coefficientから決める。

浮動scalarは `{"encoding":"ieee754-binary64-hex","value":"…"}` としてfloat.hex()の値を保存する。有理数文字列や整数Lと混同しない。NaN/Inf、未対応型、underflowはSTOP。B lawのbase theta0/theta2、B_totalも同じencodingで保存し、shot/Nは生成しない。

通常recordはschema_version/kind/producer_case/payloadのenvelope。F1 input-only handoffだけはI executor互換のroot specification/arms/rational_frames/catalog_atoms/literal_prices/score_intervalsを持つ専用schemaで保存する。価格mapは空、statusはINPUT_ONLY_PRICE_AND_F1_ENVIRONMENT_UNBOUND。F1の実行packetではない。

## Fresh namespaceと件数

`artifacts/pr_algorithm_improvement_phase2d_binding/2026-10-23/<approved_run_id>/` をtemplateとして登録する。実run IDは今回null、output directoryも作らない。future承認時に一意に指定し、親dirは専用領域へ準備してからgateで確認する。

成功時はscience11record（physical/catalog2、arm4、law5）、IO4record（source/env summary、F1 handoff、binding_summary）の計15payload、11attempt＋11completed、manifest、STOPで計39JSON、run.log1本を保存する。logは開始1行、attempt/completion22行、完了1行の24行。失敗はその時点までのimmutable filesとSTOP/errorを残し、同名を上書きしない。error textは最大8192文字。scientific runへのwall/CPU/RAM capではない。

exact bytes容量は未生成係数の桁数/mapサイズに依存する。Bで各保存bytesと合計をmanifestに記録する。巨大F1 event行列やF2 IRをBで生成しない。F1のcase range descriptorはLと整数ordinal範囲だけで、eventやtrajectoryを列挙しない。
