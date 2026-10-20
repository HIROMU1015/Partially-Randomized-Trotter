# 512件correctness契約の正式採択

ユーザーの2026-10-20指示を根拠に、固定512件correctness-only契約と一回の実科学実行を採択した。準備承認とは別の、実行まで含む明示承認である。

承認本文の完全bytesは `user_approval.json` の `approval_text` をUTF-8へ戻して取得できる。SHA256: `9a9ba20e8f5fd41a0e3d8456e8bb43f786f528351540e6e4ec996f831869d3a0`。ローカルactivation binding: `e41419e5235fd04e7c5273c26d9475d10c278033bbe9831447c29969c82138f6`。これは承認本文から生成した内部content IDであり、外部署名・認可主体・承認IDの存在を意味しない。

科学SOURCEは `76ca3c4089b201d55da0fede08e1f77733ac38a5`。直接基点は `bf86fddc0abf493d4e9e4cc40498fde72c23c194`。P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1、root256/H160、eta=rho=1e-12、epsilon_axis=1/200、primitive=1e-6、alpha_axis=1/320、familywise=1/20、A6(tau=1e-6,Kphi=J=32)を採択する。物理RZ合成保証ではない。

4方式×A/B×X/Yの各cell16 trials、S256→P1256、合計512。seed `PROPOSED_B3_CORRECTNESS_ONLY_20261020_V1`、SHA256 counter/arm/axis分離、modeはglobal IDのみに含める。accepted=0のcellがあればINCONCLUSIVE_COVERAGE、延長なし。全full Evidence、result1GiB/log64MiB共通上限。constructor/audit/angle/requeryの境界と内数規則は保存済みcall budgetを採択。新scope、一回のみ、retry0/resume false、mandatory STOP。旧claim/STOPを変更しない。

固定manifestの閉包には準備段階の説明descriptorも残す。実行可否の正本は今回の承認recordと、厳密照合された実行manifest/activation SOURCEである。新しい科学条件・sampling law・精度・予算・性能順位の採択は行わない。最終SOURCE・manifestの実値は `source_inventory.json`、`approved_manifest.json`、`preflight.json` に記録する。

Codexが同一primary agentで実装・検証する。別の独立reviewerによる検証とは呼ばない。実行開始後の不整合は修正して再実行せず、保全してSTOPする。次の担当はユーザー・GPT。
