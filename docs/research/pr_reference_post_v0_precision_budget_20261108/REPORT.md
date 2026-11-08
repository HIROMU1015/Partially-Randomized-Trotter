# 新A V0後：有限precision・role別joint予算の実行準備

2026-11-08 JST、SERVER 2。基準 **460726569c60ea4cf8f9839f2aff16f84157babc**。Branch `codex/pr-reference-post-v0-precision-budget-20261108`。

**A・Bの静的調査、最小診断helper、人工テスト、次の科学callの固定を完了しmandatory STOP。有限b表・role誤差和・新しいjoint余裕の数値評価は未実行。** 新precision／policy／保証の正式採択、改善率の推測はしていない。

## 1. 引き継ぐ結果と科学条件

[V0 REPORT](../pr_reference_post_n5b3_v0_20261108/REPORT.md)と[保存結果](../../../artifacts/pr_reference_post_n5b3_v0/2026-11-08/RESULTS.json)を引用する。全16比較で、同じb36 wordのFrobenius／spectral下界は保存局所capを超過した。この4角では同じwordのguard包絡だけを鋭くして旧局所capを満たす案は除外される。別word、role配分、joint保証、m17 RPEを除外した結果ではない。

V0-Bの中央safe upperはB′で約3.450354e-5、保存joint予算は約9.504670e-6。Referenceでは中央約1.134548e-5、予算約3.517585e-5で、他項への差が残った。いずれも保存値の引用で、実誤差の下界や全体PASSではない。旧4/16/2評価は繰り返していない。

固定背景はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A。B′4-CX、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Yを維持する。M17 q131072／r32／88 bits、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを変えない。

## 2. A：V1有限precisionの準備結果

[従来V1](../pr_reference_post_n5b3_policy_20261108/NEXT_VALIDATION_PLAN.md#v1precisiondomainの有限な構成可能性表)の選択規則を継承する。4 positive binary64 literalをp/2^eへ正確に読み、`b_max=max(36,max e)`、候補は整数36…b_max。b36は保存行を引用し、37以降だけを新しい数値実行候補にする。**今回e・b_max自体も科学入力から計算していない。**

各bの最近接index、正負対称性、選択指数、delta、Q、保存43 guardのG_known、missing指数、局所capへの残差を出すhelperを準備した。[SOURCE](../../../scripts/diagnose_pr_reference_post_v0_precision.py)は既存`round_index`の関数bodyを変更せずASTで再利用し、隔離したglobal Bだけをparameter化する。現providerをimport／変更せず、sequenceを連結しない。

既存43 wordで選択指数を全て覆えば、Q+Gのlogical上界を計算できる。Missingがあれば全上界はnullとし、`cap-Q-G_known`はmissing guardへ残る条件付き予算だけを表す。Missingは固定43 pool内の欠損であり、全archiveにwordが存在しないという結論ではない。Qを減らしてもGや選択chunk数が増え得るため、b増加だけで問題解消・T改善とはしない。

この比較は中央4 literalの構成だけを変更する診断で、非中央はC0／85の既存wordを固定する。Provider全体のBを変更してframe等まで別wordにする評価ではない。将来接続する場合は、literal一致の有限overlay等を両arm共通policyとして別途資格する必要がある。

候補の終端は4つのbinary64 targetをgridでexactに表せる精度である。しかしQ=0はlogical word誤差G=0ではなく、emitter／domain／consumer資格の解決でもない。新基本word取得やprecision採択はこの診断から自動開始しない。

## 3. B：Role別joint予算の準備結果

[CT-0 worst_counts](../../../src/trottertracks/pr_reference_ct0.py)は、期待数でなくK2全supportのapplication／run上界とcontext-max loweringを使う。一様capは、その全RZへ同じFrobenius上界を与えてspectralへ移す十分割当である。中央、frame、D等に同じ局所capを要求することは必須ではない。

SOURCEと保存template上、中央native回転はB′で2qr、Referenceでqrが安全上界。F/Gを合わせたqをさらに半減しない。K2のframe runは最大3qr、productsは各event最大2 product・各support最大2 Zを保つ。Actual SOURCE inverseは保存inverse templateを使い、forward native列のinverseへ置き換えない。

[OPTIONS](PRECISION_AND_BUDGET_OPTIONS.md)に保存の物理多重度を転記し、次の条件式へ接続した。

`sup_omega sum_role,key n_role,key(omega,axis) e_role,key + E_emitter + E_shell <= E_saved_arm`。

Roleごとの安全多重度とmax local guardを掛ける上界は使える。一方、期待多重度、価格のsigned boundary差、共通frameの価格取消しを誤差の削減へ流用しない。相関を使うならjointで同時に可能なsupport制約を別途資格する必要がある。

準備helperは、固定C0 299 recordと85 certificateの**保存logical guardだけ**からD／frames／products／CSWAPの有限key envelopeを計算する経路も備える。これは未実行の保存数値再解析である。M17へのkey移送は既存条件C・SOURCE閉包付き。中央以外の負担も表示し、scalar／vacuum／event・未coverage role・emitter／shellは必ず残項として保持する。空key集合を0にせず、SOURCE上で不存在が証明されたB′ CSWAPだけを0とする。

この表が閉じなくても、現在のmax／加法certificateの不足であって実RPE失敗とはしない。中央の再配分だけで全roleの実誤差が減るわけでもない。

## 4. 再利用範囲・残る障害

| 既存資産で準備できたこと | 残る条件 |
|---|---|
| Exact ties-to-even、saved sqrt2有理上端、43 saved guardでの有限表 | 実b表・missing・Gの変化は数値実行待ち |
| C0・85 certificateのliteral一致logical保証、actual adjoint | M17 role envelopeの数値評価、全support key移送条件C |
| 保存物理多重度・joint予算による有限の配分制約 | 未評価role、emitter／shell、supportを鋭くする追加保証 |
| 既存CT-0／N4-Cのraw disk・両軸・整数N会計 | 完成policy後の別承認RPE再会計。今回その関数は起動しない |

B′ scalarの実域外literal、Reference vacuumの保存包絡と実到達角の違い、event安全supersetとcanonical到達性を未解決のまま保持する。中央の小さい有限表はdomain問題を解消しない。詳細は[OPTIONS §3](PRECISION_AND_BUDGET_OPTIONS.md#3-domainと位相の残項)。

追加合成は未決である。現wordのまま全role予算を閉じる、又は別の保存recordを使う余地が残る。他方、選択したbのmissingや保存guard不足が、採用予定の保証・予算・provider構成で残るなら、有限追加word／guardの取得候補になる。最初から全面再合成を前提にしない。

## 5. 全18round総Tへの接続

両armへ同じarchive・precision／予算候補・policy改善・価格取得機会を与える。M17は`626*(E[T_BX]+E[T_BY])`と`630*(E[T_RX]+E[T_RY])`。共通frameが1 shotで同価格でも、この異なるshot重みでは絶対価格が残る。新T集計・solver・価格推定は実行していない。

M17のprecision資格だけで全18roundの総Tを外挿しない。完成policy、全domain／role、SOURCE→native→consumer→terminal、両軸raw誤差と保存Nの成功条件を各roundで閉じ、実行可能な強B′と比較して初めて総Tを評価する。局所精度だけの改善を無条件に採用しない。

## 6. 検証・保全・STOP

[人工test](../../../tests/test_pr_reference_post_v0_precision.py) **7件PASS**。人工角のties-to-even・符号・carry、既知／欠損guard、Qの計算、有限b規則、joint残項、role未coverage、非負の物理多重度を確認した。科学4角・保存guard・role metadataに対するhelper mainは実行していない。旧8 index／guard／160 fixture／V0も再実行していない。人工PASSは科学性能の証拠ではない。

独立したSOURCE読取を並行し、主agentのみでA/Bの準備を進めた。科学計算workerは起動せず、CPU/RAMの独自上限や共有設定変更もない。既存SOURCE／raw／result／grant／ledger／marker／STOP、他Track・他ユーザーを変更していない。正式共通指示2026-10-25、親／worktree AGENTS、PROJECT_MAPと研究概要の優先順位を適用した。

**Mandatory STOP。次担当：GPT／ユーザー。** [NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md) §2のA/B限定数値診断を実行するか判断し、実行する場合に明示承認する。承認証明書／grant発行は不要。Codexは新しい数値再解析・guard・合成・compile・RPE・T評価を開始しない。
