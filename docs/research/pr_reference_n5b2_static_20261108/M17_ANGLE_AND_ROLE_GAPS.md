# N5-B2：m17の角度域・role別の資格

2026-11-08 JST。[REPORT](REPORT.md)。下表の件数は[N5-A保存分類](../../../artifacts/pr_reference_n5a_static/2026-11-08/ROLE_METADATA.json)のm17行を引用した。新しい到達key、angle表、物理多重度、誤差和ではない。[SAVED_METADATA](../../../artifacts/pr_reference_n5b2_static/2026-11-08/SAVED_METADATA.json)へ該当行とN5-B0保存条件を引用した。

## 1. ラベルの意味

| ラベル | この表での意味 |
|---|---|
| EXISTING_RECORD_REUSABLE | 保存word／certificateが存在し、同じliteral・意味で再利用候補となる。M17の最終資格を意味しない |
| LOGICAL_GUARD_QUALIFICATION_REQUIRED | 保存logical上界をm17の保証へ接続する必要がある。新guard実行が必須という意味ではない |
| PRECISION_QUALIFICATION_REQUIRED | 量子化・選択guard和・誤差配分を含め、目標保証との適合が未確定 |
| DOMAIN_EXTENSION_REQUIRED | 現保証域では覆えない。包絡だけの超過なら、域拡張又はより強い域内証拠という分岐を残す |
| PROVIDER_POLICY_REQUIRED | M17の当該入力へwordを返す規則／保証を明示する必要がある。新アルゴリズムの実装が必須という意味ではない |
| COMPILER_CONTEXT_REQUIRED | 保存SOURCE／inverse／post-unroll templateからactual native literalへ移送するC条件を保持 |
| SOURCE_SUPPORT_UNQUALIFIED | Actual keyリスト／canonical到達性／本番経路の証拠が不足。未観測を確率0としない |

以下のroleはすべて、最終的には同じconsumer・terminal条件を必要とする。

## 2. Role別の対応

| Role | 保存証拠と現coverage | ラベル・残項 |
|---|---|---|
| Reference vacuum | 包絡32751.887444214277 rad、symbolic:vacuum_phase[m=17]。現L=82.49247654978845では包絡を覆えず、exact全keyリストなし | DOMAIN_EXTENSION_REQUIRED（又は域内証明）、PROVIDER_POLICY_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、SOURCE_SUPPORT_UNQUALIFIED、COMPILER_CONTEXT_REQUIRED |
| B′ scalar | Signed `-0x1.86e032ee41d40p+12`。Q依存の固定SOURCE項で実literalが現域外。C0／85 certificateにも対応なし | DOMAIN_EXTENSION_REQUIRED、PROVIDER_POLICY_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、COMPILER_CONTEXT_REQUIRED。Random非到達を理由に除外できない |
| B′ event sign phase／r32 | 96 positive keyの保存安全superset：C0 2、N4-B b2 1、域内未call23、域外70 | 既存3はEXISTING_RECORD_REUSABLE。全体はLOGICAL_GUARD_QUALIFICATION_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、PROVIDER_POLICY_REQUIRED、DOMAIN_EXTENSION_REQUIRED又は到達／域証拠、SOURCE_SUPPORT_UNQUALIFIED、COMPILER_CONTEXT_REQUIRED |
| Reference event位相 | 独立のcontrolled event P roleは0。Uncontrolled phaseはvacuum算術へ入る | 保存0 roleを維持しvacuum側へ接続。SOURCE_SUPPORT_UNQUALIFIEDのvacuumを省略しない |
| B′ central／r32 | ±0x1.7e4088e024c5fp-12、±0x1.fdab627b8d37ap-14。現域内だがC0／85未対応 | LOGICAL_GUARD_QUALIFICATION_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、PROVIDER_POLICY_REQUIRED、COMPILER_CONTEXT_REQUIRED |
| Reference central／r32 | ±0x1.7e4088e024c5fp-11、±0x1.fdab627b8d37ap-13。現域内だがC0／85未対応 | 同上。B′half角やr1のd0/d2価格へ同一視しない |
| DF basis frame | 両armでpositive216 key、C0132＋N4-B certificate84。SOURCE pair multisetとC下のnative閉包 | EXISTING_RECORD_REUSABLE、LOGICAL_GUARD_QUALIFICATION_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、COMPILER_CONTEXT_REQUIRED。価格共通性を誤差免除へ使わない |
| D_unfused | B′159／Reference155 positive key、全てC0という保存分類。h固定でSOURCE構造はevent非依存 | EXISTING_RECORD_REUSABLE、LOGICAL_GUARD_QUALIFICATION_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、COMPILER_CONTEXT_REQUIRED |
| D境界融合／actual inverse | 境界差B′17／Reference27、Reference actual SOURCE inverse-D supplement160 keyはC0内 | 同上。集合は重複し、signed差は負の物理gate／負のnormではない。Forward native inverseへ置換しない |
| RTE products／X/Y wrapper | 各armでroleごとpositive1 key、C0内 | EXISTING_RECORD_REUSABLE、LOGICAL_GUARD_QUALIFICATION_REQUIRED、PRECISION_QUALIFICATION_REQUIRED、COMPILER_CONTEXT_REQUIRED |
| CSWAP | B′なし、Reference positive2 keyがC0内 | Referenceは同上。CSWAP数・native構成は変更しない |

## 3. 包絡と到達性

Vacuum包絡は既存SOURCEのF/G phase state、有限event／basis／run alphabet、逐次binary64演算と最終phase和を囲った保存証拠である。**包絡>Lは、実到達角に必ず域外がある証明ではない**。現保証をそのまま使うなら広い入力域が必要だが、より強いSOURCE域証拠で同じLを使える可能性も未排除である。

B′ scalarは別で、保存された実signed literal自体が域外である。現provideはlookup前にrejectするので、このroleは既存interfaceの明確な障害となる。C0以外の旧recordが後で見つかっても、domain checkを自動変更する理由にはならない。

Event96は3rまでの保存逐次Pi0加算からの安全supersetである。70域外candidateを全てcanonical到達と認定せず、逆に保存cohortにないという理由で除外もしない。Reference vacuumのsymbolic集合とeventの有限supersetを同じcoverage件数へ合算しない。

## 4. 保存statusと実recordの違い

CT-0 KEY_COVERAGEのm17 r32中央行では、positive側statusが `STRICT_REUSE_CANDIDATE_PRECISION_UNSELECTED` でも、**records配列は空**。対応するnegative側はNUMERIC_KEY_MISSINGで、やはり空である。B′scalarも空。C0／85の対応もない。

これは新guard／再取得の結果ではなく、保存JSONの照合である。CT-0 key_mapはdefaultdictを使い、欠損keyをrecords出力でindexすると空配列が生成され、後続の同じpositive keyのmembership表示に影響し得る。従ってstatus文字列だけでwordの存在やprecision資格を認定せず、非emptyのrecord識別子・保存guard等を確認する。旧SOURCE／JSONは修正していない。

この旧CT-0 inventoryはその調査時点のpoolで、後続C0／85を含む全将来archiveの不存在証明ではない。今回の限定結論は、固定した保存pool・C0・85にr32中央の直接対応がなく、域内dyadic missとしての新しいlogical資格もまだ評価していないことである。

## 5. 次に何を判定できるか

保存r32中央の4 positive literalだけなら、既存b36・43 guardに対する選択chunkと量子化の有限診断を、合成・新guard・provider・word生成なしで設計できる。[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)に未実行の候補として固定した。

域外scalar／vacuum／eventは、この小さい中央診断で資格されない。新domain、precision、providerを選ぶ科学判断は引き続き必要。新規基本word合成の必然性は未確定。**Mandatory STOP。**
